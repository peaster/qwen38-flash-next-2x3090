#!/usr/bin/env python3
"""Peak composition of one recorded step from a torch.cuda.memory snapshot.

  memsnap.py <snapshot.pickle> [--top 30]

Replays the allocator trace recorded during the step, finds the moment of maximum net allocation relative to
the start of recording, and groups the allocations alive at that moment by their innermost frame inside vllm.
Blocks allocated before recording started are not in the trace; they are the step's persistent baseline.
"""
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict


def site(frames) -> str:
    for fr in frames or []:
        name = fr.get('filename', '')
        if '/vllm/' in name and 'qwen38_research' not in name:
            short = name.split('/vllm/', 1)[1]
            return f"{short}:{fr.get('line')} {fr.get('name')}"
    if frames:
        fr = frames[0]
        return f"{fr.get('filename')}:{fr.get('line')} {fr.get('name')}"
    return '?'


def stack(frames, depth=4) -> str:
    out = []
    for fr in frames or []:
        name = fr.get('filename', '')
        if '/vllm/' in name and 'qwen38_research' not in name:
            out.append(f"{name.split('/vllm/', 1)[1]}:{fr.get('line')} {fr.get('name')}")
        if len(out) == depth:
            break
    return ' <- '.join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('snapshot')
    p.add_argument('--top', type=int, default=30)
    a = p.parse_args()
    with open(a.snapshot, 'rb') as f:
        snap = pickle.load(f)
    traces = [t for t in snap['device_traces'] if t]
    events = traces[0]
    live = {}
    net = peak = 0
    peak_live = {}
    unmatched_free = 0
    for ev in events:
        act, addr, size = ev['action'], ev['addr'], ev['size']
        if act == 'alloc':
            live[addr] = ev
            net += size
            if net > peak:
                peak = net
                peak_live = dict(live)
        elif act == 'free_requested':
            if addr in live:
                del live[addr]
                net -= size
            else:
                unmatched_free += size
    by_site = defaultdict(lambda: [0, 0])
    by_stack = defaultdict(lambda: [0, 0])
    for ev in peak_live.values():
        s = site(ev.get('frames'))
        by_site[s][0] += ev['size']
        by_site[s][1] += 1
        by_stack[stack(ev.get('frames'))][0] += ev['size']
        by_stack[stack(ev.get('frames'))][1] += 1
    mib = 1 << 20
    reserved = sum(seg['total_size'] for seg in snap['segments'])
    allocated = sum(b['size'] for seg in snap['segments'] for b in seg['blocks']
                    if b['state'] == 'active_allocated')
    print(f'events={len(events)} peak_net={peak / mib:.1f} MiB  live_at_peak={len(peak_live)} '
          f'freed_preexisting={unmatched_free / mib:.1f} MiB')
    print(f'at dump: reserved={reserved / mib:.1f} MiB allocated={allocated / mib:.1f} MiB')
    print(f'\nlive at peak by allocation site (top {a.top}):')
    for s, (b, n) in sorted(by_site.items(), key=lambda kv: -kv[1][0])[:a.top]:
        print(f'{b / mib:9.1f} MiB {n:5d}  {s}')
    print('\nby stack:')
    for s, (b, n) in sorted(by_stack.items(), key=lambda kv: -kv[1][0])[:a.top]:
        print(f'{b / mib:9.1f} MiB {n:5d}  {s}')


if __name__ == '__main__':
    main()
