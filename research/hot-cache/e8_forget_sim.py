#!/usr/bin/env python3
"""E8 offline check: LRFU that forgets an evicted expert's score (the runtime's one-key-per-slot form) vs the
simulated LRFU that keeps history, and LRU; max-rank misses per step at a fixed capacity.

  e8_forget_sim.py <run_dir> [--capacity 76] [--lam 0.95]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import lru_sim as sim

RANKS = sim.rankings()
TR = {}


def one(args):
    run, policy, rank, layer, cap, lam = args
    calls = TR[run][rank].calls[layer]
    init = sim.seed(RANKS[layer], rank, cap)
    if policy == 'lru':
        m = sim.simulate_lru(calls, init, cap, rank)
    else:
        m = sim.simulate_scored(calls, init, cap, rank, kind='lrfu', lam=lam, keep_history=(policy == 'lrfu-keep'))
    return m[30:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--capacity', type=int, default=76)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--policies', default='lru,lrfu-keep,lrfu-forget')
    a = p.parse_args()
    TR[a.run] = {r: sim.load_trace(a.run, r) for r in (0, 1)}
    with ProcessPoolExecutor(a.workers) as pool:
        for policy in a.policies.split(','):
            ks = [(a.run, policy, r, l, a.capacity, a.lam) for r in (0, 1) for l in range(48)]
            res = list(pool.map(one, ks, chunksize=4))
            m0, m1 = np.array(res[:48]), np.array(res[48:])
            print(f'{policy:12s} cap {a.capacity} lam {a.lam}: max-rank {np.maximum(m0, m1).sum(0).mean():7.2f}', flush=True)


if __name__ == '__main__':
    main()
