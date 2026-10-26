#!/usr/bin/env python3
"""E4: can the CPU compute LRU misses faster than PCIe can copy them?

Runs inside the serving image (needs the cpu_experts extension and the checkpoint at /model):

  python3 /research/e4_cpu_bench.py --trace-dir /research/runs/e0b-hooks-on --capacity 76 \
      --cpus0 1-7 --cpus1 9-15 --steps 400

Two processes (one per EP rank) replay, step by step and layer by layer, the experts that missed on their rank
under the runtime LRU at --capacity (simulated from the recorded traces, exact per E1), computing them with
cpu_moe.layer_forward from the checkpoint's GPTQ tensors (page cache). A barrier after each layer mimics the
per-layer EP collective. Reports the per-layer max-rank latency, to compare with the measured PCIe copy cost.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

H = 2560


def parse_cpus(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def miss_jobs(trace_dir, rank, capacity, steps, skip):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import lru_sim as sim
    ranks = sim.rankings()
    t = sim.load_trace(trace_dir, rank)
    # Replay the LRU to find, per call, which ids missed (first occurrences that were not resident).
    jobs = []  # [step][layer] -> (ids int32 [T, 10] with non-miss/non-local ids = -1, weights)
    per_layer = []
    for layer in range(48):
        calls = t.calls[layer]
        init = sim.seed(ranks[layer], rank, capacity)
        slots = list(init)
        ages = [capacity - i for i in range(capacity)]
        clock = capacity
        where = {g: i for i, g in enumerate(slots)}
        rows = []
        for k, ids in enumerate(calls[:skip + steps]):
            ids = ids.tolist()
            requested = set(ids)
            protected = [s in requested for s in slots]
            seen, missed = set(), set()
            for idx, g in enumerate(ids):
                if not (0 <= g < 512) or g // 256 != rank:
                    seen.add(g)
                    continue
                slot = where.get(g)
                if slot is not None:
                    ages[slot] = clock + idx + 1
                elif g not in seen:
                    best, victim = 1 << 40, 0
                    for i in range(capacity):
                        a = 0x7FFFFFFF if protected[i] else ages[i]
                        if a < best:
                            best, victim = a, i
                    where.pop(slots[victim], None)
                    slots[victim] = g
                    where[g] = victim
                    ages[victim] = clock + idx + 1
                    protected[victim] = True
                    missed.add(g)
                seen.add(g)
            clock += len(ids)
            if k >= skip:
                arr = np.array(ids, dtype=np.int32).reshape(-1, 10)
                mask = np.isin(arr, list(missed)) if missed else np.zeros_like(arr, dtype=bool)
                rows.append((np.where(mask, arr, -1).astype(np.int32), int(len(missed))))
        per_layer.append(rows)
    n = min(len(r) for r in per_layer)
    return [[per_layer[l][s] for l in range(48)] for s in range(n)]


def worker(rank, args, barrier, results):
    import torch
    os.sched_setaffinity(0, {parse_cpus(args.main_cpus[rank])[0]})
    sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import cpu_experts
    ext = cpu_experts.build_extension()
    cpus = parse_cpus(args.cpus[rank])
    ext.init(len(cpus), cpus)
    if args.slices > 1:
        ext.set_slices({2: [0, 384, 640], 3: [0, 256, 512, 640], 5: [0, 128, 256, 384, 512, 640]}[args.slices])
    os.environ['QWEN38_CPU_EXPERTS_MODEL'] = '/model'
    svc = cpu_experts._Service.__new__(cpu_experts._Service)
    svc._maps = {}
    wm = json.load(open('/model/model.safetensors.index.json'))['weight_map']
    table = torch.zeros(48, 512, 6, dtype=torch.int64)
    for l in range(48):
        for e in range(rank * 256, rank * 256 + 256):
            prefix = f'model.language_model.layers.{l}.mlp.experts.{e}.'
            for i, (proj, kind) in enumerate(cpu_experts._PARTS):
                table[l, e, i] = svc._addr('/model', wm, prefix + proj + '.' + kind)
    skip = torch.zeros(512, dtype=torch.uint8)
    jobs = miss_jobs(args.trace_dir, rank, args.capacity, args.steps, args.skip)
    x = torch.randn(4, H, dtype=torch.float32) * 0.05
    out = torch.zeros(4, H, dtype=torch.float32)
    scratch = torch.zeros(4 * 1 << 20, dtype=torch.float32)
    w = torch.full((4, 10), 0.1, dtype=torch.float32)
    # Warm the page cache: compute every expert of this rank once (touches all its pages).
    t0 = time.perf_counter()
    for l in range(48):
        for e in range(rank * 256, rank * 256 + 256):
            ext.bench_expert(table, l, e, 1, 1)
    print(f'rank {rank}: page cache warm in {time.perf_counter() - t0:.1f}s; {len(jobs)} steps', flush=True)
    lat = np.zeros((len(jobs), 48))
    nmiss = np.zeros((len(jobs), 48), dtype=np.int32)
    barrier.wait()
    for s, step in enumerate(jobs):
        for l, (ids, n) in enumerate(step):
            t0 = time.perf_counter()
            if n:
                ext.layer_forward(table, l, x, torch.from_numpy(ids), w, skip, out, scratch)
            lat[s, l] = time.perf_counter() - t0
            nmiss[s, l] = n
            barrier.wait()
    results[rank] = (lat, nmiss)


def main():
    import multiprocessing as mp
    p = argparse.ArgumentParser()
    p.add_argument('--trace-dir', required=True)
    p.add_argument('--capacity', type=int, default=76)
    p.add_argument('--cpus0', default='1-7')
    p.add_argument('--cpus1', default='9-15')
    p.add_argument('--main0', default='0')
    p.add_argument('--main1', default='8')
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--skip', type=int, default=100)
    p.add_argument('--slices', type=int, default=1, choices=(1, 2, 3, 5))
    p.add_argument('--out')
    a = p.parse_args()
    # One CCD per rank; the process's own thread runs on a core outside its pool.
    a.cpus = [a.cpus0, a.cpus1]
    a.main_cpus = [a.main0, a.main1]
    ctx = mp.get_context('fork')
    barrier = ctx.Barrier(2)
    manager = ctx.Manager()
    results = manager.dict()
    procs = [ctx.Process(target=worker, args=(r, a, barrier, results)) for r in (0, 1)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    lat0, n0 = results[0]
    lat1, n1 = results[1]
    lat = np.maximum(lat0, lat1)
    per_step_ms = lat.sum(1) * 1e3
    pcie_ms = np.maximum(n0, n1).sum(1) * 0.176
    print(f'steps={len(per_step_ms)} misses/step rank0={n0.sum(1).mean():.1f} rank1={n1.sum(1).mean():.1f} '
          f'max-rank={np.maximum(n0, n1).sum(1).mean():.1f}')
    print(f'CPU, max-rank per layer, summed per step: mean {per_step_ms.mean():.2f} ms, '
          f'median {np.median(per_step_ms):.2f} ms')
    print(f'PCIe copy model (0.176 ms per max-rank miss): mean {pcie_ms.mean():.2f} ms')
    for k in range(0, 7):
        sel = np.maximum(n0, n1) == k
        if sel.any():
            print(f'  layers with max-rank misses={k}: CPU {lat[sel].mean() * 1e6:7.1f} us  (n={sel.sum()})')
    if a.out:
        np.savez_compressed(a.out, lat0=lat0, lat1=lat1, n0=n0, n1=n1)


if __name__ == '__main__':
    main()
