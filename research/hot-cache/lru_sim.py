#!/usr/bin/env python3
"""Offline replay of the runtime's per-layer expert LRU on recorded traces.

A trace is the sequence of LRU calls one EP rank made for one MoE layer: the flattened top-k global expert ids
of each call, in the order the runtime saw them (hooks/qwen38_research.py). Routing does not depend on the cache,
so a trace can be replayed against any capacity or policy.

Policies:
  lru      exact model of _update_lru_expert_map_kernel (and lru_map.update_kernel, which preserves it)
  opt      Belady's MIN on whole calls: evict the unprotected resident whose next use is farthest away.
           A lower bound on misses for that capacity with demand fetching; no real policy can beat it.
  static   the seeded hot set never changes (VLLM_WNA16_DYNAMIC_LRU=0)
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RANKINGS = REPO / 'configs/static_hot_cache_rankings.json'
NUM_LAYERS = 48
LOCAL = 256  # experts per EP rank, linear placement: rank r owns [256 r, 256 r + 255]
INT_MAX = 0x7FFFFFFF


@dataclass
class Trace:
    """Calls of one rank: per layer, a list of int arrays (ids in runtime order) plus recorded misses."""
    device: int
    calls: list[list[np.ndarray]] = field(default_factory=lambda: [[] for _ in range(NUM_LAYERS)])
    misses: list[list[int]] = field(default_factory=lambda: [[] for _ in range(NUM_LAYERS)])
    num_ids: list[list[int]] = field(default_factory=lambda: [[] for _ in range(NUM_LAYERS)])
    labels: list[tuple[str, int, int]] = field(default_factory=list)  # (label, first call, end call) on layer 0


def load_trace(run_dir: str | Path, device: int) -> Trace:
    """Concatenate all ring dumps of one device in dump order."""
    files = glob.glob(os.path.join(run_dir, f'lru-dev{device}-*.npz'))
    dumps = []
    for path in files:
        with np.load(path) as z:
            dumps.append((int(z['count'].max()), path, z['count'].copy(), z['last_count'].copy(),
                          z['ids'].copy(), z['meta'].copy(), int(z['ring'])))
    dumps.sort(key=lambda d: d[0])
    trace = Trace(device)
    expected = None
    for _, path, count, last, ids, meta, ring in dumps:
        label = re.sub(rf'^lru-dev{device}-|\.npz$', '', os.path.basename(path))
        if expected is not None and not np.array_equal(last, expected):
            raise ValueError(f'{path}: dumps are not contiguous')
        expected = count
        if (count - last).max() > ring:
            raise ValueError(f'{path}: ring overflow ({int((count - last).max())} calls > {ring})')
        start0 = len(trace.calls[0])
        for layer in range(NUM_LAYERS):
            for c in range(int(last[layer]), int(count[layer])):
                pos = c % ring
                n = int(meta[layer, pos, 0])
                trace.num_ids[layer].append(n)
                trace.misses[layer].append(int(meta[layer, pos, 1]))
                trace.calls[layer].append(ids[layer, pos, :min(n, ids.shape[2])].astype(np.int64))
        trace.labels.append((label, start0, len(trace.calls[0])))
    return trace


def rankings() -> dict[int, list[int]]:
    with open(RANKINGS) as f:
        raw = json.load(f)
    return {int(k): [int(v) for v in vals] for k, vals in raw.items()}


def seed(ranking: list[int], rank: int, capacity: int) -> list[int]:
    lo, hi = rank * LOCAL, (rank + 1) * LOCAL
    seen, out = set(), []
    for g in ranking:
        if lo <= g < hi and g not in seen:
            out.append(g)
            seen.add(g)
            if len(out) == capacity:
                break
    return out


def simulate_lru(calls: list[np.ndarray], initial: list[int], capacity: int, rank: int,
                 owner: np.ndarray | None = None) -> np.ndarray:
    """Exact replay of the serial LRU kernel. Returns misses per call.

    owner: optional array[512] of the rank owning each global expert (default linear placement)."""
    if owner is None:
        owner = np.arange(512) // LOCAL
    slots = list(initial) + [-1] * (capacity - len(initial))
    ages = [capacity - i for i in range(capacity)]
    clock = capacity
    where = {g: i for i, g in enumerate(slots) if g >= 0}
    out = np.zeros(len(calls), dtype=np.int32)
    for k, ids in enumerate(calls):
        ids = ids.tolist()
        requested = set(ids)
        protected = [s in requested for s in slots]
        seen = set()
        m = 0
        for idx, g in enumerate(ids):
            if not (0 <= g < 512) or owner[g] != rank:
                seen.add(g)
                continue
            slot = where.get(g)
            new_age = clock + idx + 1
            if slot is not None:
                ages[slot] = new_age
            elif g not in seen:
                best, victim = INT_MAX + 1, 0
                for i in range(capacity):
                    a = INT_MAX if protected[i] else ages[i]
                    if a < best:
                        best, victim = a, i
                old = slots[victim]
                if old >= 0:
                    where.pop(old, None)
                slots[victim] = g
                where[g] = victim
                ages[victim] = new_age
                protected[victim] = True
                m += 1
            seen.add(g)
        clock += len(ids)
        out[k] = m
    return out


def simulate_opt(calls: list[np.ndarray], initial: list[int], capacity: int, rank: int,
                 owner: np.ndarray | None = None) -> np.ndarray:
    """Belady's MIN at call granularity."""
    if owner is None:
        owner = np.arange(512) // LOCAL
    local_calls = [sorted({g for g in ids.tolist() if 0 <= g < 512 and owner[g] == rank}) for ids in calls]
    # next_use[k][g]: index of the next call after k that requests g
    nxt: dict[int, int] = {}
    next_after = [None] * len(local_calls)
    for k in range(len(local_calls) - 1, -1, -1):
        next_after[k] = dict(nxt)  # uses after k (k excluded) for every expert seen later
        for g in local_calls[k]:
            nxt[g] = k
    # memory: next_after copies are O(calls * 256); acceptable for traces of a few thousand calls
    resident = set(initial)
    out = np.zeros(len(calls), dtype=np.int32)
    for k, req in enumerate(local_calls):
        need = [g for g in req if g not in resident]
        if need:
            fut = next_after[k]
            reqs = set(req)
            candidates = sorted((g for g in resident if g not in reqs),
                                key=lambda g: fut.get(g, 1 << 60), reverse=True)
            room = capacity - len(resident)
            evict = max(0, len(need) - room)
            for g in candidates[:evict]:
                resident.discard(g)
            resident.update(need)
        out[k] = len(need)
    return out


def simulate_static(calls, initial, capacity, rank, owner=None) -> np.ndarray:
    if owner is None:
        owner = np.arange(512) // LOCAL
    resident = set(initial)
    return np.array([len({g for g in ids.tolist() if 0 <= g < 512 and owner[g] == rank} - resident)
                     for ids in calls], dtype=np.int32)


POLICIES = {'lru': simulate_lru, 'opt': simulate_opt, 'static': simulate_static}


def simulate_scored(calls, initial, capacity, rank, owner=None, *, kind='lrfu', lam=0.95, pin=0, ranking=None,
                    keep_history=True):
    """Generic demand-fetch cache at call granularity with a pluggable eviction score (lowest score evicted).

    kind='lrfu':  score = sum over past calls that used the expert of lam ** (calls since that use)
                  (lam -> 0 is LRU, lam = 1 is LFU), kept incrementally as (value, last_update_call).
    kind='lru2':  score = call index of the second most recent use (experts used once score -1, then by last use).
    pin:          the first `pin` seeded experts are never evicted (static core + dynamic rest).
    keep_history: False forgets an evicted expert's LRFU score (the runtime keeps one key per slot, E8).
    Every expert requested in the call is protected, as in the runtime."""
    import math
    if owner is None:
        owner = np.arange(512) // LOCAL
    resident = set(initial)
    pinned = set(initial[:pin])
    # Seed: give the initial residents a small, rank-ordered history so ties resolve like the runtime seed.
    val = {g: (capacity - i) * 1e-6 for i, g in enumerate(initial)}
    upd = {g: 0 for g in initial}
    last1 = {g: -(i + 1) for i, g in enumerate(initial)}
    last2 = {}
    out = np.zeros(len(calls), dtype=np.int32)
    loglam = math.log(lam) if 0 < lam < 1 else None

    def score(g, k):
        if kind == 'lrfu':
            v = val.get(g, 0.0)
            if loglam is None:
                return v
            return v * math.exp(loglam * (k - upd.get(g, k)))
        if kind == 'lru2':
            return (last2.get(g, -1 << 40), last1.get(g, -1 << 40))
        raise ValueError(kind)

    for k, ids in enumerate(calls):
        req = [g for g in dict.fromkeys(ids.tolist()) if 0 <= g < 512 and owner[g] == rank]
        need = [g for g in req if g not in resident]
        if need:
            reqs = set(req)
            room = capacity - len(resident)
            evict = max(0, len(need) - room)
            if evict:
                cands = [g for g in resident if g not in reqs and g not in pinned]
                cands.sort(key=lambda g: score(g, k))
                for g in cands[:evict]:
                    resident.discard(g)
                    if not keep_history:
                        val.pop(g, None)
                        upd.pop(g, None)
            resident.update(need)
        out[k] = len(need)
        for g in req:
            if kind == 'lrfu':
                val[g] = score(g, k) + 1.0
                upd[g] = k
            else:
                if g in last1:
                    last2[g] = last1[g]
                last1[g] = k
    return out


POLICIES['lrfu'] = simulate_scored


def simulate_admit(calls, initial, capacity, rank, owner=None, *, lam=0.95, rule='freq', window=8, opt=False):
    """Demand cache with bypass: a non-resident requested expert always costs one transfer, but it is inserted
    only if admitted; otherwise it is computed from host memory and the cache is left unchanged.

    rule='freq':   admit if its decayed frequency (incl. this use) exceeds the would-be victim's (TinyLFU-like)
    rule='window': admit if it was also requested in one of the previous `window` calls
    opt=True:      Belady with bypass: admit only if its next use comes before the farthest resident's next use."""
    import math
    if owner is None:
        owner = np.arange(512) // LOCAL
    resident = set(initial)
    val = {g: (capacity - i) * 1e-6 for i, g in enumerate(initial)}
    upd = {g: 0 for g in initial}
    last = {}
    loglam = math.log(lam)
    out = np.zeros(len(calls), dtype=np.int32)
    local_calls = [list(dict.fromkeys(g for g in ids.tolist() if 0 <= g < 512 and owner[g] == rank)) for ids in calls]
    if opt:
        nxt_use = []
        nxt: dict[int, int] = {}
        for k in range(len(local_calls) - 1, -1, -1):
            nxt_use.append(dict(nxt))
            for g in local_calls[k]:
                nxt[g] = k
        nxt_use.reverse()

    def score(g, k):
        return val.get(g, 0.0) * math.exp(loglam * (k - upd.get(g, k)))

    for k, req in enumerate(local_calls):
        need = [g for g in req if g not in resident]
        out[k] = len(need)
        for g in req:
            val[g] = score(g, k) + 1.0
            upd[g] = k
        if need:
            reqs = set(req)
            for g in need:
                if len(resident) < capacity:
                    resident.add(g)
                    continue
                cands = [h for h in resident if h not in reqs]
                if not cands:
                    continue
                if opt:
                    fut = nxt_use[k]
                    victim = max(cands, key=lambda h: fut.get(h, 1 << 60))
                    admit = fut.get(g, 1 << 60) < fut.get(victim, 1 << 60)
                else:
                    victim = min(cands, key=lambda h: score(h, k))
                    if rule == 'freq':
                        admit = score(g, k) > score(victim, k)
                    else:
                        admit = last.get(g, -1 << 40) >= k - window
                if admit:
                    resident.discard(victim)
                    resident.add(g)
        for g in req:
            last[g] = k
    return out


def simulate_split(calls, initial, capacity, rank, owner=None, *, share=0.35, insert='pcie'):
    """Hybrid miss path: per call, ceil(share * m) of the m misses are copied over PCIe (and inserted, LRU), the
    rest are computed on the CPU. insert='pcie': only copied experts enter the cache; 'all': CPU-computed experts
    are also inserted (asynchronously, e.g. after the step), so the LRU sees every miss.
    Returns int array [calls, 2]: (pcie copies, cpu experts)."""
    if owner is None:
        owner = np.arange(512) // LOCAL
    slots = list(initial) + [-1] * (capacity - len(initial))
    ages = [capacity - i for i in range(capacity)]
    clock = capacity
    where = {g: i for i, g in enumerate(slots) if g >= 0}
    out = np.zeros((len(calls), 2), dtype=np.int32)
    import math
    for k, ids in enumerate(calls):
        ids = ids.tolist()
        req = list(dict.fromkeys(g for g in ids if 0 <= g < 512 and owner[g] == rank))
        for g in req:
            if g in where:
                ages[where[g]] = clock + 1 + ids.index(g)
        missed = [g for g in req if g not in where]
        n_pcie = math.ceil(share * len(missed))
        admit = missed if insert == 'all' else missed[:n_pcie]
        protected = set(req)
        for g in admit:
            best, victim = None, None
            for i in range(capacity):
                if slots[i] in protected:
                    continue
                if best is None or ages[i] < best:
                    best, victim = ages[i], i
            if victim is None:
                break
            if slots[victim] >= 0:
                where.pop(slots[victim], None)
            slots[victim] = g
            where[g] = victim
            ages[victim] = clock + 1 + ids.index(g)
            protected.add(g)
        clock += len(ids)
        out[k] = (n_pcie, len(missed) - n_pcie)
    return out
