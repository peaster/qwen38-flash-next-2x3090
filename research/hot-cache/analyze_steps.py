#!/usr/bin/env python3
"""H1: regress decode step time on the runtime's own per-step miss counts.

  analyze_steps.py <run_dir> [<run_dir> ...]

For every dump window, the last len(step_tokens) LRU calls of each layer belong to the window's small steps, in
order. step_ms[i] is the GPU time from small step i to i+1. Only 4-token steps (MTP3 verify) followed by a
4-token step are used, so prefill boundaries and flush requests drop out.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np


def load(run_dir):
    rows = []  # (step_ms, misses rank0 per layer [48], misses rank1 per layer [48])
    per_dev = {}
    for dev in (0, 1):
        dumps = []
        for path in glob.glob(os.path.join(run_dir, f'lru-dev{dev}-*.npz')):
            z = np.load(path)
            dumps.append((int(z['count'].max()), os.path.basename(path).split('-', 2)[2], z))
        dumps.sort(key=lambda d: d[0])
        per_dev[dev] = dumps
    for (c0, label, z0), (c1, _, z1) in zip(per_dev[0], per_dev[1]):
        tokens = z0['step_tokens']
        ms = z0['step_ms']
        n = len(tokens)
        if n < 3:
            continue
        ring = int(z0['ring'])
        miss = []
        for z in (z0, z1):
            count = z['count']
            m = np.zeros((n, 48), dtype=np.int32)
            for layer in range(48):
                idx = (np.arange(count[layer] - n, count[layer])) % ring
                m[:, layer] = z['meta'][layer, idx, 1]
            miss.append(m)
        for i in range(n - 1):
            if tokens[i] == 4 and tokens[i + 1] == 4:
                rows.append((ms[i], miss[0][i], miss[1][i], label))
    return rows


def main():
    for run_dir in sys.argv[1:]:
        rows = load(run_dir)
        if not rows:
            print(run_dir, 'no steps')
            continue
        ms = np.array([r[0] for r in rows])
        m0 = np.array([r[1] for r in rows])
        m1 = np.array([r[2] for r in rows])
        maxrank = np.maximum(m0, m1).sum(1)
        total = (m0 + m1).sum(1)
        r0 = m0.sum(1)
        keep = ms < np.percentile(ms, 99)  # drop the slowest 1% (scheduling hiccups)
        out = [f'{os.path.basename(run_dir)}: steps={len(ms)} step_ms mean={ms.mean():.2f} median={np.median(ms):.2f}',
               f'  misses/step: rank0={r0.mean():.1f} rank1={m1.sum(1).mean():.1f} max-rank={maxrank.mean():.1f}']
        for name, x in (('max-rank', maxrank), ('total', total), ('rank0', r0)):
            A = np.vstack([x[keep], np.ones(keep.sum())]).T
            (b, a), res, *_ = np.linalg.lstsq(A, ms[keep], rcond=None)
            pred = A @ np.array([b, a])
            r2 = 1 - ((ms[keep] - pred) ** 2).sum() / ((ms[keep] - ms[keep].mean()) ** 2).sum()
            out.append(f'  step_ms = {a:.2f} + {b * 1000:.1f} us x {name}   R^2={r2:.3f}')
        print('\n'.join(out))


if __name__ == '__main__':
    main()
