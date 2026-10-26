#!/usr/bin/env python3
"""E1 analysis: validate the LRU simulator against the runtime, then compute miss curves offline.

  analyze_traces.py <run_dir> --capacity 76 [--out results.json]

Cost model: every target MoE layer ends in an EP collective, so a decode step waits, layer by layer, for the
rank with more misses. The step metric is therefore sum over layers of max over ranks of misses ("max-rank").
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import lru_sim as sim

CAPS = list(range(48, 129, 4))


def _job(args):
    policy, calls, ranking, cap, rank = args
    return sim.POLICIES[policy](calls, sim.seed(ranking, rank, cap), cap, rank)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('run_dir', type=Path)
    p.add_argument('--capacity', type=int, required=True, help='hot slots the run used')
    p.add_argument('--out', type=Path)
    p.add_argument('--policies', default='lru,opt,static')
    a = p.parse_args()
    ranks = sim.rankings()
    traces = {r: sim.load_trace(a.run_dir, r) for r in (0, 1)}
    report: dict = {'run_dir': str(a.run_dir), 'capacity': a.capacity}

    # Trace health.
    for r, t in traces.items():
        n = np.array(t.num_ids[0])
        report[f'rank{r}_calls'] = len(n)
        report[f'rank{r}_num_ids_hist'] = {int(k): int(v) for k, v in zip(*np.unique(n, return_counts=True))}
        report[f'rank{r}_truncated_calls'] = int(sum((np.array(t.num_ids[l]) > 40).sum() for l in range(48)))
        report['labels'] = t.labels
    print({k: v for k, v in report.items() if k != 'labels'})

    with ProcessPoolExecutor() as pool:
        # 1. Validation at the run's own capacity.
        jobs = [('lru', traces[r].calls[l], ranks[l], a.capacity, r) for r in (0, 1) for l in range(48)]
        sims = list(pool.map(_job, jobs, chunksize=1))
        val = {}
        for i, (r, l) in enumerate((r, l) for r in (0, 1) for l in range(48)):
            rec = np.array(traces[r].misses[l])
            s = sims[i]
            val[(r, l)] = (int(rec.sum()), int(s.sum()), float((rec == s).mean()) if len(rec) else 1.0)
        exact = np.mean([v[2] for v in val.values()])
        rec_total = sum(v[0] for v in val.values())
        sim_total = sum(v[1] for v in val.values())
        print(f'validation: per-call exact match {exact:.4%}; recorded misses {rec_total}, simulated {sim_total}')
        report['validation'] = {'per_call_exact_fraction': exact, 'recorded_misses': rec_total,
                                'simulated_misses': sim_total,
                                'worst_layers': sorted(([r, l, *v] for (r, l), v in val.items()),
                                                       key=lambda x: x[4])[:5]}

        # 2. Curves: misses per call for every (policy, rank, layer, capacity).
        policies = a.policies.split(',')
        keys = [(pol, r, l, c) for pol in policies for r in (0, 1) for l in range(48) for c in CAPS]
        results = list(pool.map(_job, [(pol, traces[r].calls[l], ranks[l], c, r) for pol, r, l, c in keys],
                                chunksize=4))
    curves = {k: v for k, v in zip(keys, results)}
    calls = len(traces[0].calls[0])
    report['curves'] = {}
    for pol in policies:
        rows = []
        for c in CAPS:
            per_rank = [sum(curves[(pol, r, l, c)].sum() for l in range(48)) / calls for r in (0, 1)]
            max_rank = sum(np.maximum(curves[(pol, 0, l, c)], curves[(pol, 1, l, c)]).sum()
                           for l in range(48)) / calls
            rows.append({'capacity': c, 'rank0': per_rank[0], 'rank1': per_rank[1], 'max_rank': max_rank})
        report['curves'][pol] = rows
        print(f'\n{pol}: misses per step (sum over 48 layers)')
        for row in rows:
            print(f"  cap {row['capacity']:3d}: rank0 {row['rank0']:7.2f}  rank1 {row['rank1']:7.2f}  "
                  f"max-rank {row['max_rank']:7.2f}")
    # Per-layer misses per call at the run capacity, both ranks.
    report['per_layer_lru'] = [[float(curves[('lru', r, l, a.capacity)].mean()) if a.capacity in CAPS else None
                                for l in range(48)] for r in (0, 1)]
    report['per_layer_curves_lru'] = {f'{r}:{l}': [float(curves[('lru', r, l, c)].mean()) for c in CAPS]
                                      for r in (0, 1) for l in range(48)}
    report['caps'] = CAPS
    if a.out:
        a.out.write_text(json.dumps(report, indent=1, default=str))


if __name__ == '__main__':
    main()
