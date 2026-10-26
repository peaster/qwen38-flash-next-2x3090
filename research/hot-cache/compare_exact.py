#!/usr/bin/env python3
"""E7: compare exact.json probes within and across runs.

  compare_exact.py <control_run_dir> <candidate_run_dir>

For each prompt, reports how many prompt/output logprobs are bitwise equal and the largest absolute difference:
control repeat 0 vs 1 (the control's own nondeterminism), candidate 0 vs 1, and control vs candidate per repeat.
"""
import json
import sys
from pathlib import Path


def load(run):
    rows = json.loads((Path(run) / 'exact.json').read_text())
    return {(r['repeat'], r['kind'], r['length']): r for r in rows}


def cmp(a, b):
    out = []
    for key in ('prompt_logprobs', 'output_logprobs'):
        x, y = a[key], b[key]
        pairs = [(p, q) for p, q in zip(x, y) if p is not None and q is not None]
        same = sum(p == q for p, q in pairs)
        diff = max((abs(p - q) for p, q in pairs), default=0.0)
        out.append(f'{key.split("_")[0]} {same}/{len(pairs)} eq, max|d| {diff:.3g}')
    out.append('tokens ' + ('same' if a['output_tokens'] == b['output_tokens'] else 'DIFFER'))
    return '; '.join(out)


def main():
    ctl, cand = load(sys.argv[1]), load(sys.argv[2])
    prompts = sorted({(k, n) for _, k, n in ctl})
    for kind, n in prompts:
        print(f'== {kind}:{n}')
        print('  ctl 0 vs 1 :', cmp(ctl[(0, kind, n)], ctl[(1, kind, n)]))
        print('  cand 0 vs 1:', cmp(cand[(0, kind, n)], cand[(1, kind, n)]))
        for r in (0, 1):
            print(f'  ctl vs cand r{r}:', cmp(ctl[(r, kind, n)], cand[(r, kind, n)]))


if __name__ == '__main__':
    main()
