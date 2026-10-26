"""E4b: decompose CPU per-layer latency: dispatch overhead vs expert compute (distinct experts, no L3 reuse)."""
import json, os, sys, time
import numpy as np, torch
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import cpu_experts
ext = cpu_experts.build_extension()
os.sched_setaffinity(0, {0})
svc = cpu_experts._Service.__new__(cpu_experts._Service); svc._maps = {}
wm = json.load(open('/model/model.safetensors.index.json'))['weight_map']
table = torch.zeros(48, 512, 6, dtype=torch.int64)
for l in range(48):
    for e in range(256):
        for i, (proj, kind) in enumerate(cpu_experts._PARTS):
            table[l, e, i] = svc._addr('/model', wm, f'model.language_model.layers.{l}.mlp.experts.{e}.{proj}.{kind}')
for l in range(48):
    for e in range(256):
        ext.bench_expert(table, l, e, 1, 1)
skip = torch.zeros(512, dtype=torch.uint8)
x = torch.randn(4, 2560) * 0.05; out = torch.zeros(4, 2560); scratch = torch.zeros(4 << 20); w = torch.full((4, 10), 0.1)
rng = np.random.default_rng(0)
def run(threads, cpus, n_experts, reps=600):
    ext.init(threads, cpus)
    lat = []
    for r in range(reps):
        l = int(rng.integers(48)); es = rng.choice(256, n_experts, replace=False)
        ids = torch.full((4, 10), -1, dtype=torch.int32)
        for j, e in enumerate(es): ids[j % 4, j // 4] = int(e)
        t0 = time.perf_counter(); ext.layer_forward(table, l, x, ids, w, skip, out, scratch); lat.append(time.perf_counter() - t0)
    return np.median(lat) * 1e6
ids0 = torch.full((4, 10), -1, dtype=torch.int32)
ext.init(7, list(range(1, 8)))
t0 = time.perf_counter()
for _ in range(2000): ext.layer_forward(table, 0, x, ids0, w, skip, out, scratch)
print(f'empty layer_forward: {(time.perf_counter() - t0) / 2000 * 1e6:.1f} us')
# single-thread expert compute, distinct experts
ts = []
for r in range(300):
    l = int(rng.integers(48)); e = int(rng.integers(256))
    ts.append(ext.bench_expert(table, l, e, 1, 1))
print(f'bench_expert 1 thread, T=1, distinct (median incl. 1 warm rep): {np.median(ts) * 1e6:.1f} us')
SLICES = {1: [0, 640], 2: [0, 384, 640], 3: [0, 256, 512, 640], 5: [0, 128, 256, 384, 512, 640]}
for sl in (1, 2, 3, 5):
    ext.set_slices(SLICES[sl])
    for threads, cpus in ((7, list(range(1, 8))), (15, list(range(1, 16)))):
        print(f'slices={sl} threads={threads:2d}: ' + '  '.join(f'{n} exp {run(threads, cpus, n):7.1f} us' for n in (1, 2, 3, 4, 8)), flush=True)
