#!/usr/bin/env python3
"""E8 check: the runtime LRU kernel (with the e8 files mounted) against lru_sim on recorded traces.

Runs inside the serving image on one GPU, with QWEN38_LRU_LRFU_LAMBDA set or unset:

  python3 /research/e8_kernel_test.py --trace-dir /research/runs/e0b-hooks-on --capacity 76 --layers 0,17,40

Feeds every recorded call of the chosen layers and ranks through update_kernel (the parallel kernel with its
serial fallback, as at runtime) and compares the per-call miss counts with simulate_lru (lambda unset) or
simulate_scored(kind='lrfu') (lambda set).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lru_sim as sim  # noqa: E402

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E402
    compressed_tensors_moe_wna16 as wna16,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.lru_map import (  # noqa: E402
    update_kernel,
)
from vllm.triton_utils import triton  # noqa: E402


def run_kernel(calls, initial, capacity, rank, device):
    source_map = torch.full((512,), -1, dtype=torch.int32)
    source_map[rank * 256:(rank + 1) * 256] = torch.arange(256, dtype=torch.int32)
    cache_map = torch.full((512,), -1, dtype=torch.int32)
    for slot, g in enumerate(initial):
        cache_map[g] = slot
    source_map, cache_map = source_map.to(device), cache_map.to(device)
    slot_ids = torch.tensor(initial, dtype=torch.int32, device=device)
    if wna16.LRFU:
        ages = wna16.lrfu_initial_ages(initial, device)
        clock = torch.zeros(1, dtype=torch.int32, device=device)
    else:
        ages = torch.arange(capacity, 0, -1, dtype=torch.int32, device=device)
        clock = torch.tensor([capacity], dtype=torch.int32, device=device)
    miss_ids = torch.empty(160, dtype=torch.int32, device=device)
    miss_slots = torch.empty(160, dtype=torch.int32, device=device)
    counts = []
    for ids in calls:
        ids_t = torch.from_numpy(ids.astype(np.int32)).to(device)
        n = ids_t.numel()
        update_kernel[(1,)](ids_t, source_map, cache_map, slot_ids, ages, clock, miss_ids, miss_slots,
                            num_ids=n, global_num_experts=512, capacity=capacity,
                            id_block=triton.next_power_of_2(n), capacity_block=triton.next_power_of_2(capacity),
                            num_warps=4)
        counts.append(miss_ids[:n].clone())
    torch.cuda.synchronize(device)
    return np.array([int((c >= 0).sum()) for c in counts], dtype=np.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace-dir', required=True)
    p.add_argument('--capacity', type=int, default=76)
    p.add_argument('--layers', default='0,17,40')
    p.add_argument('--lam', type=float, default=float(os.environ.get('QWEN38_LRU_LRFU_LAMBDA', '0')))
    a = p.parse_args()
    device = torch.device('cuda:0')
    ranks = sim.rankings()
    total = agree = 0
    for rank in (0, 1):
        trace = sim.load_trace(a.trace_dir, rank)
        for layer in map(int, a.layers.split(',')):
            calls = [c for c in trace.calls[layer] if c.size <= 160]
            initial = sim.seed(ranks[layer], rank, a.capacity)
            got = run_kernel(calls, initial, a.capacity, rank, device)
            if a.lam > 0:
                want = sim.simulate_scored(calls, initial, a.capacity, rank, kind='lrfu', lam=a.lam)
            else:
                want = sim.simulate_lru(calls, initial, a.capacity, rank)
            same = int((got == want).sum())
            first = int(np.argmax(got != want)) if same < len(got) else -1
            print(f'rank {rank} layer {layer}: calls {len(got)} agree {same} ({100 * same / len(got):.3f}%) '
                  f'misses kernel {got.sum()} sim {want.sum()} first diff {first}', flush=True)
            total += len(got)
            agree += same
    print(f'lambda={a.lam} LRFU={bool(wna16.LRFU)}: {agree}/{total} calls agree ({100 * agree / total:.3f}%)')


if __name__ == '__main__':
    main()
