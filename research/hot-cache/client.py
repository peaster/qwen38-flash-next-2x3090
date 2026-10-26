#!/usr/bin/env python3
"""Workload client for the hot-cache study.

  client.py wait  <run_dir> [--log server.log]
  client.py suite <run_dir> <suite> [--repeat N]

Every request is a greedy, forced-length, streamed /v1/completions call made through
scripts/benchmark_serving.measure_run (exact token accounting, unique cache salt). Before and after each
request the client writes <run_dir>/trigger so the research hooks dump their LRU ring with that label; the
dump happens at the start of the next step, so a one-token flush request ends every suite.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
import benchmark_serving as bs  # noqa: E402

BASE = 'http://127.0.0.1:8000'
MODEL = 'Qwen3.8-Flash-Next'
TIMEOUT = 3600.0

# Public, mild prompts covering different domains. Outputs are not kept.
CHATS = {
    'code': 'Implement a persistent (immutable) red-black tree in Python with insert, delete and in-order '
            'iteration. Include type hints, docstrings and a pytest test suite with property-based tests.',
    'math': 'Prove that there are infinitely many primes congruent to 3 mod 4. Then determine, with proof, all '
            'positive integers n such that n^2 + 2n + 2 divides n^4 + 4.',
    'story': 'Write a short story (about 1,200 words) about a lighthouse keeper on a remote island who finds a '
             'message in a bottle written in her own handwriting.',
    'explain': 'Explain how TCP congestion control works. Compare Reno, CUBIC and BBR, including how each reacts '
               'to packet loss and to bufferbloat, and when you would choose each.',
    'translate': 'Translate the following paragraph into French, German, Spanish and Japanese, then explain three '
                 'idioms that were hard to translate: "The early bird catches the worm, but the second mouse gets '
                 'the cheese. When it rains it pours, so we decided to bite the bullet and finish the project '
                 'before the deadline, even if it meant burning the midnight oil."',
    'json': 'Design a JSON Schema for a library management system (books, members, loans, fines). Then produce '
            '15 realistic sample records as a single JSON document that validates against the schema.',
}

# E8 held-out set: domains and wording not used by E1/E3 traces (the cache policies were fitted on those).
HELDOUT = {
    'sql': 'Design a PostgreSQL schema for a bike-sharing service (stations, bikes, riders, trips, maintenance). '
           'Write the DDL with constraints and indexes, then ten analytical queries with window functions, and '
           'explain the query plan you expect for the two most expensive ones.',
    'physics': 'Derive the period of a simple pendulum for large amplitudes using elliptic integrals, then compare '
               'it numerically with the small-angle approximation for amplitudes of 10, 45, 90 and 170 degrees.',
    'recipe': 'Plan a week of vegetarian dinners for a family of four on a modest budget. For each day give a recipe '
              'with quantities, preparation steps, and a shopping list grouped by store section.',
    'rust': 'Write a Rust command-line tool that deduplicates files in a directory tree by content hash, using '
            'rayon for parallel hashing. Include error handling, unit tests and a short README.',
    'history': 'Explain the causes and consequences of the 1848 revolutions in Europe, country by country, and '
               'discuss why most of them failed in the short term.',
    'poem': 'Write a sequence of twelve short poems, one for each month, about a small mountain village, each in '
            'a different poetic form, with a one-line note naming the form.',
}

SUITES = {
    # E0: hook overhead; the matched short-decode shape.
    'short': [('repo-chat', 128, 256)],
    # E1: routing traces over diverse decode content plus one long-context decode.
    'traces': [('repo-chat', 128, 2048)] + [(f'chat:{k}', None, 1536) for k in CHATS] +
              [('repo-chat', 131072, 2048)],
    # Long-context prefill and decode, the 10-03 protocol shape.
    'long': [('repo-chat', 131072, 2048)],
    # E2: one long prefill for the memory snapshot.
    'prefill': [('repo-chat', 131072, 128)],
    # Full-context capacity probe (AGENTS.md: required before a 256K claim).
    'full': [('repo-chat', 262016, 128)],
    # E8: held-out decode for cache policies fitted on the E1/E3 traces.
    'heldout': [(f'held:{k}', None, 1536) for k in HELDOUT],
    # E3: decode timing at each hot size.
    'decode': [('repo-chat', 128, 2048), ('chat:code', None, 1536), ('chat:explain', None, 1536)],
}


def scrape_metrics() -> dict[str, float]:
    """All vLLM counters and histogram sums/counts from /metrics."""
    with urllib.request.urlopen(BASE + '/metrics', timeout=30) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith('#'):
            continue
        m = re.match(r'^(vllm:[a-zA-Z0-9_:]+(?:_total|_sum|_count))(\{[^}]*\})? ([0-9.eE+-]+)$', line)
        if m:
            labels = re.sub(r'(engine|model_name)="[^"]*",?', '', m.group(2) or '').replace('{}', '')
            out[m.group(1) + labels] = out.get(m.group(1) + labels, 0.0) + float(m.group(3))
    return out


def host_counters() -> dict[str, int]:
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    mem = {k: int(v.split()[0]) for k, v in (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    return {'pswpin': int(vm['pswpin']), 'pswpout': int(vm['pswpout']),
            'mem_available_kib': mem['MemAvailable'], 'swap_free_kib': mem['SwapFree']}


def derived(result: dict, prompt_len: int, out: int) -> dict:
    """Prefill, decode and inter-chunk latency figures from one streamed run."""
    chunks = [c for c in result.get('sse_chunks', []) if c['output_token_ids_count'] > 0]
    times = [c['elapsed_seconds'] for c in chunks]
    gaps = [b - a for a, b in zip(times, times[1:])]
    toks = [c['output_token_ids_count'] for c in chunks[1:]]
    per_token = sorted(g / t for g, t in zip(gaps, toks) if t)
    q = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))] if xs else None
    sg = sorted(gaps)
    ttft = result['ttft_seconds']
    return {
        'prefill_tokens_per_second': prompt_len / ttft if ttft else None,
        'decode_tokens_per_second': result['api_observed_decode_estimate_tokens_per_second'],
        'tokens_per_chunk_mean': statistics.mean(toks) if toks else None,
        'chunk_gap_ms': {k: (q(sg, f) * 1e3 if sg else None) for k, f in
                         (('p50', .5), ('p90', .9), ('p99', .99), ('max', 1.0))},
        'itl_per_token_ms': {k: (q(per_token, f) * 1e3 if per_token else None) for k, f in
                             (('p50', .5), ('p90', .9), ('p99', .99))},
        'chunks_with_tokens': len(chunks),
    }


def write_trigger(run_dir: Path, label: str) -> None:
    tmp = run_dir / 'trigger.tmp'
    tmp.write_text(label + '\n')
    tmp.replace(run_dir / 'trigger')


def build_prompt(kind: str, length: int | None) -> list[int]:
    if kind == 'repo-chat':
        source, _ = bs.prompt_source('repo-chat')
        prompt, _ = bs.repo_chat_prompt(BASE, MODEL, TIMEOUT, source, length)
        return prompt
    if kind.startswith('chat:'):
        return bs.tokenize_chat(BASE, MODEL, TIMEOUT, CHATS[kind[5:]])
    if kind.startswith('held:'):
        return bs.tokenize_chat(BASE, MODEL, TIMEOUT, HELDOUT[kind[5:]])
    raise SystemExit(f'unknown prompt kind {kind}')


EXACT_PROMPTS = [('repo-chat', 128), ('repo-chat', 20000), ('chat:math', None)]


def exact(run_dir: Path, repeat: int) -> None:
    """E7 exactness probe: greedy prompt and output logprobs on fixed prompts (20K = three 8K chunks).

    Prompt logprobs only up to 2,048 tokens: an 8K chunk would need ~8 GB of full-vocabulary logits.

    Each request gets a fresh cache salt so the whole prompt is prefilled. Only token ids and logprob values are
    stored (exact.json); comparing two runs of one server measures its own nondeterminism.
    """
    runs = []
    for r in range(repeat):
        for kind, length in EXACT_PROMPTS:
            prompt = build_prompt(kind, length)
            body = {'model': MODEL, 'prompt': prompt, 'max_tokens': 32, 'temperature': 0.0, 'logprobs': 1,
                    'prompt_logprobs': 0 if len(prompt) <= 2048 else None, 'cache_salt': f'e7-{time.time_ns()}', 'ignore_eos': True}
            req = urllib.request.Request(f'{BASE}/v1/completions', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                choice = json.load(resp)['choices'][0]
            plp = [None if d is None else next(iter(d.values()))['logprob'] for d in choice.get('prompt_logprobs') or []]
            lp = choice['logprobs']
            runs.append({'repeat': r, 'kind': kind, 'length': len(prompt),
                         'output_tokens': lp['tokens'], 'output_logprobs': lp['token_logprobs'],
                         'prompt_logprobs': plp})
            print(f'exact {r} {kind}:{len(prompt)} prompt_lp={len(plp)} out={len(lp["tokens"])}', flush=True)
    (run_dir / 'exact.json').write_text(json.dumps(runs))


def wait(run_dir: Path, log: Path | None) -> None:
    start = time.time()
    while True:
        try:
            with urllib.request.urlopen(BASE + '/v1/models', timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(5)
    print(f'ready after {time.time() - start:.0f}s', flush=True)
    if log is not None:
        while True:
            text = log.read_text(errors='replace')
            # Profiles without a CPU PLE table (hot-only, NVMe mmap) never prefault.
            if 'PLE prefault complete' in text or 'QWEN38_PLE_PREFAULT=0' in text or 'CPU experts (rank' in text:
                break
            if time.time() - start > 1800:
                print('PLE prefault not reported; continuing', flush=True)
                break
            time.sleep(5)


def suite(run_dir: Path, name: str, repeat: int, warmup: int) -> None:
    items = SUITES[name]
    prompts = {(kind, length): build_prompt(kind, length) for kind, length, _ in items}
    results = []
    write_trigger(run_dir, f'{name}-start')
    gpu_log = open(run_dir / f'gpu-{name}.csv', 'a')
    sampler = subprocess.Popen(
        ['nvidia-smi', '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm,'
         'temperature.gpu,pcie.link.gen.current,pcie.link.width.current', '--format=csv,noheader', '-lms', '500'],
        stdout=gpu_log, stderr=subprocess.DEVNULL)
    plan = [('warmup', i) for i in range(warmup)] + [('measured', i) for i in range(repeat)]
    for phase, rep in plan:
        for kind, length, out in items:
            prompt = prompts[(kind, length)]
            label = f'{name}-{phase}{rep}-{kind.replace(":", "_")}-{len(prompt)}-{out}'
            write_trigger(run_dir, f'pre-{label}')
            m0, h0 = scrape_metrics(), host_counters()
            result = bs.measure_run(BASE, MODEL, prompt, out, TIMEOUT, phase, rep)
            m1, h1 = scrape_metrics(), host_counters()
            write_trigger(run_dir, label)
            result['label'] = label
            result['derived'] = d = derived(result, len(prompt), out)
            result['server_metrics_delta'] = {k: v - m0.get(k, 0.0) for k, v in m1.items() if v != m0.get(k, 0.0)}
            result['host_delta'] = {'pswpin': h1['pswpin'] - h0['pswpin'], 'pswpout': h1['pswpout'] - h0['pswpout'],
                                    'mem_available_kib_end': h1['mem_available_kib'],
                                    'swap_free_kib_end': h1['swap_free_kib']}
            md = result['server_metrics_delta']
            drafts = md.get('vllm:spec_decode_num_draft_tokens_total')
            accepted = md.get('vllm:spec_decode_num_accepted_tokens_total')
            d['mtp_acceptance'] = accepted / drafts if drafts else None
            d['preemptions'] = md.get('vllm:num_preemptions_total', 0.0)
            results.append(result)
            print(f"{label}: ttft={result['ttft_seconds']:.2f}s prefill={d['prefill_tokens_per_second']:.0f} tok/s "
                  f"decode={d['decode_tokens_per_second'] or 0:.2f} tok/s "
                  f"chunk p50/p99={d['chunk_gap_ms']['p50'] or 0:.1f}/{d['chunk_gap_ms']['p99'] or 0:.1f} ms "
                  f"tok/chunk={d['tokens_per_chunk_mean'] or 0:.2f} accept={d['mtp_acceptance'] or 0:.3f} "
                  f"preempt={d['preemptions']:.0f} swapin={result['host_delta']['pswpin']} "
                  f"status={result['status']}", flush=True)
            out_path = run_dir / f'suite-{name}.json'
            out_path.write_text(json.dumps(results, indent=1))
    # Flush: the dump for the last label happens at the first step of this request.
    flush = bs.tokenize(BASE, MODEL, TIMEOUT, 'flush')
    bs.measure_run(BASE, MODEL, flush, 1, TIMEOUT, 'flush', 0)
    sampler.terminate()
    gpu_log.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    w = sub.add_parser('wait')
    w.add_argument('run_dir', type=Path)
    w.add_argument('--log', type=Path)
    s = sub.add_parser('suite')
    s.add_argument('run_dir', type=Path)
    s.add_argument('suite', choices=sorted(SUITES))
    s.add_argument('--repeat', type=int, default=1)
    s.add_argument('--warmup', type=int, default=0)
    x = sub.add_parser('exact')
    x.add_argument('run_dir', type=Path)
    x.add_argument('--repeat', type=int, default=2)
    args = parser.parse_args()
    if args.cmd == 'wait':
        wait(args.run_dir, args.log)
    elif args.cmd == 'exact':
        exact(args.run_dir, args.repeat)
    else:
        suite(args.run_dir, args.suite, args.repeat, args.warmup)


if __name__ == '__main__':
    main()
