#!/usr/bin/env python3
"""E8: measured vs predicted effect of the cache policies on held-out prompts.

  e8_analyze.py [--runs e8-lru,e8-lrfu,e8-alloc,e8-both,e8-lru2]

Measured: per run, mean max-rank misses per 4-token verify step and mean step GPU time (from the recorder dumps,
same selection as analyze_steps.py), plus the step-time fit. Predicted: the control run's (e8-lru) traces replayed
offline under each policy; the relative change of the max-rank miss mean is compared with the measured one.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import analyze_steps
import lru_sim as sim

RUNS = 'runs'
SIZES = [int(v) for v in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'overrides/e8-alloc.env'))
         .read().split('=', 1)[1].split(',')]
POLICIES = {'e8-lru': (0.0, False), 'e8-lrfu': (0.95, False), 'e8-alloc': (0.0, True), 'e8-both': (0.95, True),
            'e8-lru2': (0.0, False)}
RANKS = sim.rankings()
TR = {}


def replay(args):
    lam, alloc, rank, layer = args
    cap = SIZES[layer] if alloc else 76
    calls = TR[rank].calls[layer]
    init = sim.seed(RANKS[layer], rank, cap)
    if lam:
        return sim.simulate_scored(calls, init, cap, rank, kind='lrfu', lam=lam)[30:]
    return sim.simulate_lru(calls, init, cap, rank)[30:]


def measured(run):
    rows = analyze_steps.load(os.path.join(RUNS, run))
    ms = np.array([r[0] for r in rows])
    mx = np.array([np.maximum(r[1], r[2]).sum() for r in rows])
    keep = ms < np.percentile(ms, 99)
    b, a = np.polyfit(mx[keep], ms[keep], 1)
    return len(ms), mx.mean(), ms[keep].mean(), a, b * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='e8-lru,e8-lrfu,e8-alloc,e8-both,e8-lru2')
    a = p.parse_args()
    runs = a.runs.split(',')
    base = measured(runs[0])
    print('measured (4-token verify steps):')
    print('  run        steps  max-rank misses  step ms (trim 1%)   fit')
    meas = {}
    for run in runs:
        n, mx, ms, ia, sl = measured(run)
        meas[run] = (mx, ms)
        print(f'  {run:9s} {n:6d}  {mx:7.2f} ({100 * (mx / base[1] - 1):+5.1f}%)  {ms:6.2f} ({100 * (ms / base[2] - 1):+5.1f}%)'
              f'   {ia:.2f} + {sl:.1f} us x miss')
    for r in (0, 1):
        TR[r] = sim.load_trace(os.path.join(RUNS, runs[0]), r)
    print(f'predicted from {runs[0]} traces (replay, all calls):')
    with ProcessPoolExecutor(8) as pool:
        ref = None
        for run in runs:
            lam, alloc = POLICIES[run]
            res = list(pool.map(replay, [(lam, alloc, r, l) for r in (0, 1) for l in range(48)], chunksize=4))
            mx = np.maximum(np.array(res[:48]), np.array(res[48:])).sum(0).mean()
            ref = ref or mx
            print(f'  {run:9s} max-rank {mx:7.2f} ({100 * (mx / ref - 1):+5.1f}%)')


if __name__ == '__main__':
    main()
