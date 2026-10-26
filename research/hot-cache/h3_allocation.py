#!/usr/bin/env python3
"""H3: non-uniform per-layer slot allocation at a fixed total, fitted on one trace set and scored on another.

  h3_allocation.py <fit_run_dir> <eval_run_dir> [--total-per-layer 76]

Greedy marginal allocation on the fit traces' per-layer LRU curves (mean of both ranks, steps of 4 slots, same
slot count on both GPUs). Constraint: each layer keeps 64..128 slots (H3_FLOOR, default 64; runtime geometry
checks allow 16..128); the last 8 layers keep >= 64 so stream staging can still borrow 512 rows. The chosen allocation and the uniform one are then replayed exactly on the
eval traces with the max-rank metric.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import lru_sim as sim

import os
FLOOR = int(os.environ.get('H3_FLOOR', '64'))
CAPS = list(range(min(FLOOR, 64), 129, 4))
RANKS = sim.rankings()
TR = {}


def curve(args):
    run, r, l, c = args
    return sim.simulate_lru(TR[run][r].calls[l], sim.seed(RANKS[l], r, c), c, r)[30:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('fit')
    p.add_argument('eval')
    p.add_argument('--total-per-layer', type=int, default=76)
    a = p.parse_args()
    for run in {a.fit, a.eval}:
        TR[run] = {r: sim.load_trace(run, r) for r in (0, 1)}
    with ProcessPoolExecutor() as pool:
        keys = [(a.fit, r, l, c) for r in (0, 1) for l in range(48) for c in CAPS]
        fit = dict(zip(keys, pool.map(curve, keys, chunksize=8)))
        mean = {(l, c): (fit[(a.fit, 0, l, c)].mean() + fit[(a.fit, 1, l, c)].mean()) / 2 for l in range(48) for c in CAPS}
        budget = 48 * a.total_per_layer
        # Floor per layer; the last 8 layers keep >= 64 each so stream staging can still borrow 512 rows.
        alloc = [FLOOR] * 40 + [max(FLOOR, 64)] * 8
        while sum(alloc) + 4 <= budget:
            gains = [(mean[(l, alloc[l])] - mean[(l, alloc[l] + 4)], l) for l in range(48) if alloc[l] + 4 <= 128]
            g, l = max(gains)
            alloc[l] += 4
        uniform = [a.total_per_layer] * 48
        evals = {}
        for name, al in (('uniform', uniform), ('fitted', alloc)):
            ks = [(a.eval, r, l, al[l]) for r in (0, 1) for l in range(48)]
            res = list(pool.map(curve, ks, chunksize=4))
            m0 = np.array(res[:48]); m1 = np.array(res[48:])
            evals[name] = (m0.sum(0).mean(), m1.sum(0).mean(), np.maximum(m0, m1).sum(0).mean())
    print('allocation (layer: slots):', ' '.join(f'{l}:{s}' for l, s in enumerate(alloc)))
    print(f'total {sum(alloc)} (uniform {sum(uniform)}); last-8 rows {sum(alloc[-8:])}')
    for name, (r0, r1, mx) in evals.items():
        print(f'{name:8s} on eval: rank0 {r0:6.2f} rank1 {r1:6.2f} max-rank {mx:6.2f}')
    u, f = evals['uniform'][2], evals['fitted'][2]
    print(f'max-rank change: {100 * (f - u) / u:+.1f}%')


if __name__ == '__main__':
    main()
