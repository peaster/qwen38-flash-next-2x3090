# Expert cache policy and prefill-memory trims on the x8 host — October 25

Three runtime changes, measured on the October 3 host with one locally built image (this tree's
`docker/Dockerfile`, image ID in [`summary.json`](summary.json); not a published image):

1. **Prefill-memory trims (always on).** The PLE forward drops its embeddings, normalized key/query and value
   as soon as they are consumed, and the decoder loop hands each layer its only reference to the stream state.
   No operation changes; an 8,192-token chunk needs 258 MiB less transient memory.
2. **LRFU expert replacement (`QWEN38_LRU_LRFU_LAMBDA`, opt-in).** The per-layer GPU expert cache evicts the
   expert with the lowest exponentially decayed use count instead of the least recently used one.
3. **Per-layer cache sizes (`QWEN38_HOT_CACHE_LAYER_SIZES`, opt-in).** The same total slots, split unevenly:
   layer 0 misses about ten times as often as layer 31.

[`configs/fast-256k-prefill.env`](../../configs/fast-256k-prefill.env) now enables 2 and 3 at the same 76 slots
per layer and GPU; the new optional [`configs/fast-128k-prefill.env`](../../configs/fast-128k-prefill.env)
spends the smaller 128K KV pool on 92 slots.

## Why the expert cache

On this host decode is bound by expert copies. With CUDA events around every MTP verify step, the step time
fits 17.0–17.3 ms plus 176–179 µs per expert miss on the slower of the two GPUs (summed over the 48 MoE layers),
R² ≥ 0.997 for 64–96 slots. One miss is one 2.42 MiB expert over PCIe 4.0 ×8; at 76 slots about half of a verify
step is copies. Every change here reduces misses or makes room for more slots; none changes a kernel's math.
The two policies were chosen offline: the recorded routing of real requests was replayed through an exact model
of the runtime cache (99.986% of calls identical), then the runtime kernel was checked against that model call by
call (100.000% with the policies off, 99.95–99.99% with LRFU).

## Results (released image, one session each, same day)

| Configuration | Decode, held-out prompts (6) | Decode, `decode` suite (3) | Decode after 131K (2) | 131K prefill, input tok/s (TTFT) | 262,016 + 128 | Preemptions |
|---|---:|---:|---:|---|---|---:|
| Control: 256K profile without the two policy lines (76 slots, LRU) | 74.03 | 68.20 | 73.29 | 4,182 / 4,178 (31.3 / 31.4 s) | 3,978 / 4,145 tok/s | 0 |
| **[`fast-256k-prefill.env`](../../configs/fast-256k-prefill.env)**: + LRFU + per-layer sizes | **78.57** (+6.1%) | **74.57** (+9.3%) | **80.45** (+9.8%) | 4,198 / 4,200 (31.2 / 31.2 s) | 3,968 / 4,157 tok/s | 0 |
| **[`fast-128k-prefill.env`](../../configs/fast-128k-prefill.env)**: 135,168 window, 92 slots, both policies | **89.78** (+21.3%) | **86.47** (+26.8%) | **86.95** (+18.6%) | 3,981 / 4,342 (32.9 / 30.2 s) | — (128K window) | 0 |

Decode is the reciprocal mean of the API-observed time per output token. MTP acceptance was 0.51–0.53 per draft
token in all three. Held-out prompts were not used to fit either policy (they are listed in `summary.json`).

Mechanism checks from the research sessions (CUDA events and the cache's own miss counters):

- Same VRAM, held-out prompts, two LRU sessions against LRFU / per-layer / both: max-rank misses −6.6% / −7.5% /
  −14.2%, verify step −2.9% / −4.2% / −7.1%. The offline replay had predicted each within about 2 points.
- Same day against two control sessions: the 256K profile cut misses 12–13% and the verify step 6%; the 128K
  profile cut misses 37–40% and the verify step 19–20%.
- Free GPU memory at the worst 8K chunk: the October 3 profile on the previous image (no trims) 193–207 MiB,
  256K profile 321–335 MiB, 128K profile 281–295 MiB. The trims more than pay for the per-layer sizes' own
  overhead.

## Exactness of the trims

The runtime is not bitwise deterministic (approximate QSA, MTP batches): two control runs differ by up to 2 nats
in prompt logprobs on 128-token prompts (median 0.005). Against that, control versus trims: greedy tokens
identical in all 12 comparisons, |Δ logprob| median 0.0076 and p95 0.77 (control vs itself 0.0051 and 0.81).
The trims only drop Python references, so they are exact by construction.

## What did not work

- **hot80 at 256K.** With the trims it passed 262,016 + 128 once at ~100 MiB free; with the policies it then ran
  out of memory on a 320 MiB request while 786 MiB sat reserved but unallocated in the caching allocator.
  A ~100 MiB margin is not robust; the shipped profile keeps 76 slots and gains headroom instead.
- **hot96 at 128K with the policies**, same 320 MiB request; hot100 failed even without them.
- **LRFU that forgets an evicted expert's score** kept less than half of the gain; the runtime keeps one score
  per expert (512 extra int32 per layer).

## Caveats

- One session per configuration. Single requests vary by about ±10% with MTP acceptance.
- Per-layer sizes cost 50–170 MiB of non-PyTorch GPU memory (each distinct slot count has its own Humming
  configuration and kernels) and compile time: about 8 minutes of LRU-kernel compilation the first time the
  image starts with an empty `JIT_CACHE_DIR`, and 4.8–5.7 s to first token on a session's first short request
  (control 1.9 s). Later starts reuse the cache.
- The sizes were fitted on this workload's routing (`repo-chat` and chat prompts); other workloads may prefer a
  different split. They do not change the model's outputs, only which experts stay on the GPU.
- These are measurements on one x8 host; they do not change the published x16 claims.
