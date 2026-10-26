#!/usr/bin/env python3
"""E9 prediction: replay the held-out E8 control traces under F5 and both candidate cache configurations."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import lru_sim as sim

RANKS = sim.rankings()
TR = {r: sim.load_trace('runs/e8-lru', r) for r in (0, 1)}


def sizes(path):
    return [int(v) for v in open(path).read().split('=', 1)[1].split(',')]


CONFIGS = {'F5 (hot76 LRU)': (0.0, [76] * 48), '256K candidate': (0.95, sizes('overrides/e9-alloc80.env')),
           '128K candidate': (0.95, sizes('overrides/e9-alloc96.env'))}


def replay(args):
    lam, caps, rank, layer = args
    calls, cap = TR[rank].calls[layer], caps[layer]
    init = sim.seed(RANKS[layer], rank, cap)
    if lam:
        return sim.simulate_scored(calls, init, cap, rank, kind='lrfu', lam=lam)[30:]
    return sim.simulate_lru(calls, init, cap, rank)[30:]


with ProcessPoolExecutor(8) as pool:
    base = None
    for name, (lam, caps) in CONFIGS.items():
        res = list(pool.map(replay, [(lam, caps, r, l) for r in (0, 1) for l in range(48)], chunksize=4))
        mx = np.maximum(np.array(res[:48]), np.array(res[48:])).sum(0).mean()
        base = base or mx
        step = 17.2 + 0.177 * mx
        print(f'{name:16s} max-rank {mx:6.2f} ({100 * (mx / base - 1):+5.1f}%)  step {step:5.2f} ms')
