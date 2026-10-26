#!/usr/bin/env python3
"""E9 diagnosis: GPU memory held by the LRU update kernels the startup warmup compiles for each slot count.

  e9_module_mem.py <comma-separated capacities>

Runs update_kernel for every warmup count (10, 20, ... <= capacity) of each distinct capacity, as lru_warmup does,
and reports the driver's free-memory drop (loaded CUDA modules live outside PyTorch's allocator).
"""
import os
import sys

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import compressed_tensors_moe_wna16  # noqa: F401  (import order: lru_map imports it)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.lru_map import update_kernel
from vllm.triton_utils import triton

caps = sorted({int(v) for v in sys.argv[1].split(',')})
dev = torch.device('cuda:0')
torch.cuda.init()
torch.zeros(1, device=dev)
torch.cuda.synchronize()
free0 = torch.cuda.mem_get_info()[0]
launches = 0
for cap in caps:
    for count in range(10, cap + 1, 10):
        ids = torch.full((count,), -1, dtype=torch.int32, device=dev)
        src = torch.full((512,), -1, dtype=torch.int32, device=dev)
        hot = torch.full((512,), -1, dtype=torch.int32, device=dev)
        slots = torch.arange(cap, dtype=torch.int32, device=dev)
        ages = torch.zeros(cap + 512, dtype=torch.int32, device=dev)
        clock = torch.zeros(1, dtype=torch.int32, device=dev)
        mi, ms = torch.empty_like(ids), torch.empty_like(ids)
        update_kernel[(1,)](ids, src, hot, slots, ages, clock, mi, ms, num_ids=count, global_num_experts=512,
                            capacity=cap, id_block=triton.next_power_of_2(count), capacity_block=128, num_warps=4)
        launches += 1
torch.cuda.synchronize()
torch.cuda.empty_cache()
free1 = torch.cuda.mem_get_info()[0]
print(f'LRFU={os.environ.get("QWEN38_LRU_LRFU_LAMBDA", "0")} capacities={len(caps)} specializations={launches} '
      f'driver free drop {(free0 - free1) / 2**20:.1f} MiB')
