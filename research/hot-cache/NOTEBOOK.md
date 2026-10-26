# Hot expert cache research notebook

Started 2026-10-25. Host: 2× RTX 3090 (PCIe 4.0 ×8, NVLink), RTX 5060 Ti holding the PLE table, 124 GiB RAM.
Baseline profile: `.env` = F5 (`configs/fast-256k-prefill.env` + PLE GPU): 8,192-token chunks, hot76,
`QWEN38_EMBED_UVA=1`, stream staging, async scheduling, MTP3.

The Log section is append-only. Hypotheses and predictions are written before the experiment that tests them.
The sections above the Log (status, hypotheses and verdicts, experiment index, plan) are kept current.

## Status (2026-10-25, end of day 1, after E6–E8)

1. **Decode on this host is PCIe-copy-bound** (E3, confirmed again in E8): verify-step time = 17.0–17.3 ms +
   0.176–0.179 ms × (misses on the slower EP rank, summed over layers), R² ≥ 0.997. Every lever below works only
   by changing that miss count.
2. **The offline LRU replay is exact** (E1: 99.986% of calls) and **predicts runtime policy changes on unseen
   prompts within ~2 points** (E8).
3. **Cache policy in the runtime (E8, held-out prompts, same VRAM):** LRFU λ=0.95 −6.6%, per-layer allocation
   −7.5%, both −14.2% max-rank misses; verify step −7.1%; API decode +7.0% (72.5 → 77.6 tok/s at 128-token
   context). The runtime kernel matches the simulator call for call (E8 kernel check). Not yet in the overlay.
4. **Exact prefill-memory trims (E7)** cut the 8K-chunk transient by 258 MiB with no op changes; greedy output is
   identical and logprob differences match the control's own nondeterminism. With them **hot80 fits at 256K**
   with F5's margin and passes 262,016+128 (E7b).
5. **Config frontier at 128K (E6):** with 8K chunks, hot96 fits (hot100 OOMs in the GDN chunk): ~4,300 tok/s
   prefill and 85 tok/s decode after a 131K prompt, versus ~70 for hot76 at the same context.
6. Ruled out: rank placement (H4), moving MoE temporaries into borrowed hot pages (H5), the static hot-only CPU
   design on this host (E5), static pinning and admission control (E1c).
7. MTP acceptance on these prompts is 0.40–0.76 per draft token, far below the historical 86–90%. Every step pays
   a fixed ~17 ms, so acceptance is a separate decode lever outside this study.

8. **E9/E9c (combined candidates vs F5, same day):** hot80 at 256K and hot96 at 128K OOM'd on allocator
   fragmentation (margins ~100 MiB are not robust; E7b revised). Margin-keeping candidates pass: 256K (hot76 +
   trims + LRFU + per-layer sizes) −6% verify step, +6–9% decode, 321 MiB headroom (F5 193); 128K (hot92 + same)
   −19 to −20% step, +16–27% decode, unchanged prefill. Per-layer sizes cost 50–170 MiB outside PyTorch and
   first-start compiles; a warmup bug (capacity_block) is fixed in `e8/lru_warmup.py`.

Promoted 2026-10-25 (see the Log): trims always on; LRFU and per-layer sizes on in `configs/fast-256k-prefill.env`;
optional `configs/fast-128k-prefill.env`. Release validation: 256K +6–10% decode, 128K +19–27%. Next: decide
between H6 (link pooling) and H7 (CPU hybrid) from the new baseline. See Plan.

## Question

Each hot slot costs 121 MB per GPU (one 2.42 MiB INT4 expert × 48 MoE layers). Decode rises with slot count
(10-03 on this host: hot72 68.5, hot76 71.3, hot84 76.2 tok/s at 128+256). But 8,192-token prefill chunks only
fit with hot76. Can the hot cache be organized better, either by spending the same VRAM more effectively or by
borrowing it back during prefill, so that this host gets both ~4,000 tok/s prefill and hot84-or-better decode?

## Background facts (from code; E1–E3 later confirmed the LRU semantics and the cost of a miss)

- LRU semantics (`_update_lru_expert_map_kernel`, `lru_map.update_kernel`): one LRU per layer per EP rank over
  that rank's 256 local experts. One call per target MoE layer per step with ≤16 tokens
  (`VLLM_WNA16_STATIC_HOT_CACHE_MAX_TOKENS`). The ids are `topk_ids` flattened token-major. Only the first
  occurrence counts. Every expert requested in the call is protected. A miss evicts the oldest unprotected slot
  and copies the expert's packed rows and scales from the pinned host source (`_gather_lru_rows`) before the
  GEMM. With `QWEN38_STAGE_OVERLAP=1` and exactly 4 tokens, the down-projection copy overlaps the gate/up GEMM.
- Steps with more than 16 tokens never touch the LRU. During staged prefill (≥ `QWEN38_STREAM_STAGE_MIN_TOKENS`)
  every layer's complete 256-expert pool is DMA'd into one of two slots borrowed from the hot pages of the last
  8 MoE layers, and those 8 layers are restored from the immutable source before the next small step. The hot
  pages of the other 40 layers (~7.2 GiB per GPU at hot76) are neither read nor written during staged prefill.
- Routing does not depend on the hot cache: the hot rows are byte-identical copies, so `topk_ids` are the same for
  any slot count (up to runtime nondeterminism). One routing trace can be replayed against any cache geometry.

## Hypotheses

- **H1 (decode is miss-bound).** Decode cycle time ≈ a + b·M, where M is the number of LRU misses per cycle on
  the slower rank. b is close to one expert copy at the measured link speed (2.42 MiB / 13.4 GB/s ≈ 0.19 ms),
  minus whatever the M4 overlap hides.
- **H2 (the simulator is exact).** Replaying captured per-call `topk_ids` through a Python model of the LRU
  reproduces the runtime's miss count per call exactly. Then miss curves for any slot count, policy or layout
  can be computed offline.
- **H3 (layers are not alike).** Per-layer miss rates at a fixed slot count differ by more than 2×. A non-uniform
  allocation with the same total slots reduces total misses by ≥10% versus uniform.
- **H4 (rank imbalance).** Per-cycle miss counts differ between the two EP ranks, and the cycle waits for the
  slower one. Balancing (per-rank slot counts or placement) reduces max-rank misses by ≥10%.
- **H5 (prefill can borrow the hot cache).** The extra VRAM an 8,192-token chunk needs over the hot80 budget
  (F3 OOM: 80 MiB requested, 25 MiB free) comes from short-lived allocations inside the prefill forward whose
  total peak is ≤ the ~7 GiB of hot pages that sit idle during staged prefill. If those allocations, or enough
  of them, are served from borrowed hot pages, hot≥84 with 8,192-token chunks fits, prefill stays within 5% of
  F5, and decode matches F1 (hot84).
- **H6 (link pooling, added after E1b).** If the less-loaded rank fetches part of the busier rank's misses over its
  own PCIe link and forwards them over NVLink, the step pays roughly the per-rank mean instead of the per-step max
  (at hot76 ~71 instead of ~99 max-rank misses, ~5 ms per step).
- **H7 (LRU + CPU hybrid, added after E4).** Copying ~25% of each layer's misses over PCIe (inserted into the
  LRU) and computing the rest on a sliced CPU pool cuts the miss path from ~19.6 to ~10.3 ms per step at hot76.

### Verdicts

| | verdict | evidence |
|---|---|---|
| H1 | **supported** | E3: 177 µs per max-rank miss, R² ≥ 0.997, intercept 17.2 ms, six slot counts |
| H2 | **supported** | E1: 99.986% per-call exact; mismatches only in startup/capture calls |
| H3 | **supported in the runtime** | per-layer misses differ ~10× (layer 0 vs 31); offline −6.2 / −9.4% (below the ≥10% bar alone); E8 runtime −7.5% alone, −14.2% with LRFU on held-out prompts |
| H4 | **refuted** | E1b: systematic imbalance ~2 misses of the ~27 max-rank excess; the rest is step-level noise |
| H5 | **mechanism refuted** | E2: the 8K-chunk peak is HC/norm/PLE activations; nothing MoE-internal to move |
| H6 | untested | simulated ceiling only (E1b) |
| H7 | open | E4 prices the CPU side outside the runtime; E5 shows small bridge overhead; needs a Humming-layout CPU kernel |

## Protocol

Each experiment records: git commit plus the research tooling diff, the full `.env` and overrides, image ID,
host RAM/swap state, JIT cache state, warmup policy, and raw outputs under `runs/<experiment>/`
(`launch.txt` and `env.txt` record the image, commit and every setting). Controls are
run in the same session as treatments. Two or more long runs per configuration; below ~5%, differences are
noise (10-03 finding).

Per-request metrics (`client.py`, every suite): TTFT and prefill tok/s (input/TTFT), decode tok/s
(post-first-chunk reciprocal TPOT), SSE chunk gaps and per-token ITL (p50/p90/p99/max), tokens per chunk, MTP
acceptance, preemptions and all other vLLM counter deltas from `/metrics`, host swap-in/out and MemAvailable,
and a 0.5 s nvidia-smi sample (memory, utilization, power, SM clock, PCIe link) per suite. Standard suites:
`short` = 128+256, 1 warmup + 5 measured; `long` = 131,072+2,048, 2 measured, no warmup (10-03 protocol).

Instrumentation (`hooks/`, loaded through `PYTHONPATH` sitecustomize only when `QWEN38_RESEARCH=1`):
- LRU trace: per layer and call, the flattened `topk_ids` (int16, 40 wide) and the runtime's miss count, kept
  in a device ring buffer written inside the CUDA graph and dumped on request. It changes no runtime decision.
- Step memory: per large step, the allocator peak, reserved bytes and driver-free bytes. Optionally one
  `torch.cuda.memory` history snapshot of an N-th large step.
- Hook overhead must be measured: run E0 (control with hooks off vs on, same session) before trusting the
  timings of hooked runs.

## Experiments

| ID | Tests | Design | Status | Runs |
|---|---|---|---|---|
| E0 | instrumentation is inert | F5 hooks off vs on: short ×5 + long ×2 | done: inert | `e0a-hooks-off`, `e0b-hooks-on` (+ `e0a-hooks-off-v1`) |
| E1 | H2; miss curves | F5 + hooks, short/long/traces suites; replay in `lru_sim.py` | done: H2 supported | `e0b-hooks-on/e1-analysis.json` |
| E1b | H4 | per-layer mean vs per-step max decomposition | done: H4 refuted | offline |
| E1c | online policies | LRFU, LRU-2, pinning, admission, Belady bounds | done: best −6.4% | offline |
| E1d | H3 | greedy per-layer allocation, fit on one trace set, score on the other | done: −6.2 / −9.4% | `h3_allocation.py` |
| E2 | H5 composition | memory snapshot, 2nd 8K chunk of the 3rd 131K prompt | done: not MoE | `e0b-hooks-on/memsnap-*` |
| E3 | H1 | hot 64–96, 128K context, 4K chunks, per-step GPU timing | done: H1 supported | `e3-hot*` |
| E4 | CPU miss cost | two-rank replay of hot76 misses through `cpu_moe`, thread/slice variants; microbenchmarks | done | `e4/`, `e4b_micro.py` |
| E5 | in-situ CPU path | 64 GB hot-only profile, default and 5 slices | done: s1 held, s5 refuted | `e5-64gb-s1`, `e5-64gb-s5` |
| E6 | config frontier | ≤128K context: hot76–100 with 8,192-token chunks; prefill + decode | done: hot96 fits, +22% decode | `e6-hot*` |
| E7 | exact prefill trims | free PLE key/query early; drop the loop's stale residual; 8K peak and exactness | done: −258 MiB, exact | `e7-ctl`, `e7-trim`, `e7/` |
| E7b | max hot at 256K with trims | hot80 + trims: 262,016+128, 131K, decode | done: hot80 fits | `e7b-hot80` |
| E8 | policy in runtime | LRFU λ=0.95 and the fitted per-layer allocation; held-out prompts | done: −14.2% misses, −7.1% step | `e8-*`, `e8/` |
| E9 | combined candidates | 256K trims+hot80+policies, 128K trims+hot96+policies vs F5 ×2 | done: both OOM (fragmentation) | `e9-*` |
| E9c | margin-keeping candidates | 256K hot76+trims+policies, 128K hot92+trims+policies | done: −6% / −20% step, pass | `e9c-*` |

## Plan

Done on 2026-10-25 in the agreed order: E6, E7 (+E7b), E8. Proposed next, cheapest and most certain first:

1. ~~**E9, combined candidates.**~~ Done (E9, E9c): see the Log. 256K: trims + hot80 + LRFU + allocation refitted at 80 slots/layer; 128K: trims +
   hot96 + LRFU + allocation refitted at 96. Against F5 in the same day: `full` (256K only), `long` ×2, `decode`
   and `heldout` ×2, with hooks for misses and step time. Prediction from E3/E8: the 256K candidate cuts verify
   steps ~10% versus F5 (4 slots ≈ −3%, policies ≈ −7%).
2. ~~**Promote to the overlay**~~ Done 2026-10-25. What E9c confirmed: the `e8/` files (incl. the `lru_warmup.py` fix) (LRFU and per-layer sizes are opt-in,
   default off), the two `e7/` files (always on: exact), a config, benchmark JSON and AGENTS.md notes, per the
   repo's rules (overlay rebuilt from the measured image, SHA256SUMS updated).
3. Then decide between H6 (link pooling, ~5 ms/step simulated ceiling, now smaller since misses fell) and H7
   (CPU hybrid, ~9 ms/step predicted, needs a Humming-layout CPU kernel), using E9's new baseline.

## Files

| file | role |
|---|---|
| `hooks/sitecustomize.py`, `hooks/qwen38_research.py` | observation hooks (LRU ring recorder, step GPU timing, step memory, memory snapshot), on with `QWEN38_RESEARCH=1` |
| `launch.sh`, `session.sh` | start the image with hooks, overrides and extra vLLM args; one start→suites→stop session |
| `client.py` | workload suites (`short`, `long`, `traces`, `decode`, `prefill`) and all per-request metrics |
| `overrides/*.env` | per-experiment settings layered over `.env` |
| `lru_sim.py` | exact LRU replay, Belady bounds, static, LRFU/LRU-2/pinning, admission, CPU split |
| `analyze_traces.py`, `analyze_steps.py`, `memsnap.py`, `h3_allocation.py` | E1, E3, E2, E1d analyses |
| `e4_cpu_bench.py`, `e4b_micro.py`, `e3.sh` … `e8.sh`, `e7b.sh`, `e8-resume.sh` | CPU benchmarks and experiment drivers |
| `e7/` | E7 candidate: `model.py`, `ple_layer.py` (reference drops only) |
| `e8/` | E8 candidate: `compressed_tensors_moe_wna16.py` (LRFU, `QWEN38_HOT_CACHE_LAYER_SIZES`), `lru_map.py`, `stream_stage.py` |
| `compare_exact.py`, `e8_kernel_test.py`, `e8_forget_sim.py`, `e8_analyze.py` | E7 exactness, E8 kernel-vs-simulator, LRFU variants, E8 measured-vs-predicted |
| `runs/` | raw outputs (server logs, client JSON, LRU ring dumps, step timings, memory snapshots); gitignored, local only (~2 GB). The release-validation reports are published in `benchmarks/2026-10-25/raw/` |

Operational notes: wait loops must not `pgrep -f` a pattern that appears in their own command line (that cost one
E4 delay and killed two shells). The hot-only profile never logs `PLE prefault complete`, and `client.py wait`
now accepts its `CPU experts` line instead.

## Log

### 2026-10-25 E0 v1 (superseded): control with hooks off, decode only

`runs/e0a-hooks-off-v1`: 128+256, 1 warmup + 5: 69.92, 71.05, 71.22, 73.08, 70.75 tok/s (10-03 F5: 71.32).
This first client recorded only TTFT and decode rate, with no long-context run, ITL, acceptance or host
counters. The client was extended and E0 rerun with the full metric set; v1 is kept as an extra control point.

### 2026-10-25 E0: instrumentation is inert (supported)

`runs/e0a-hooks-off` vs `runs/e0b-hooks-on`, same image and `.env` (F5), back-to-back sessions.

| | hooks off | hooks on |
|---|---:|---:|
| 128+256 decode, reciprocal mean of 5 (range) | 71.10 (70.10–72.04) | 70.37 (67.62–74.07) |
| 128+256 MTP acceptance (mean) | 0.505 | 0.545 |
| 131K prefill tok/s | 3,972 / 4,203 | 3,934 / 4,099 |
| 131K decode tok/s | 74.34 / 73.69 | 74.08 / 71.50 |
| chunk gap p50 / p99, 131K | 34.3 / 69.7 ms | 34.2 / 68.4 ms |
| preemptions | 0 | 0 |

Decode differences are inside the run-to-run range. Prefill is 1–2% lower with hooks, consistent with the
per-large-step synchronize of `STEPMEM`. The control reproduces 10-03 F5 (3,976 / 4,196 prefill, 72.0 decode).
Hooked timings are usable for decode. Prefill comparisons must use hooks-off runs or both arms hooked.

Side observation: MTP acceptance on these workloads is 0.40–0.68 per draft token (2.2–3.0 tokens per SSE chunk),
far below the 86–90% recorded on the historical repetitive prompt. Decode speed is also an acceptance problem.

### 2026-10-25 E1: the simulator is exact (H2 supported); miss curves

`runs/e0b-hooks-on`: 7,417 LRU steps per rank (short ×6, long ×2, traces suite), 7,413 of them 4-token MTP3
verify steps. The LRU replay matches the runtime's per-call miss count on 99.986% of (rank, layer, call) triples.
Recorded total misses 1,193,585 vs simulated 1,193,559. All mismatches are in the first ~20 calls: startup and
graph-capture shapes, including one 50-id call wider than the recorder.

Misses per decode step, summed over 48 layers (`e1-analysis.json`):

| slots | LRU rank0 / rank1 | LRU max-rank | Belady optimum max-rank | static max-rank |
|---:|---:|---:|---:|---:|
| 64 | 108.6 / 98.7 | 139.9 | 77.9 | 386 |
| 76 | 84.4 / 76.5 | **111.3** | 59.5 | 341 |
| 84 | 71.5 / 64.6 | 95.7 | 49.9 | 312 |
| 96 | 55.7 / 50.3 | 76.4 | 38.5 | 273 |
| 128 | 27.8 / 25.2 | 40.9 | 19.3 | 185 |

"max-rank" is the sum over layers of the per-step maximum over the two ranks: every MoE layer ends in an EP
collective, so that is what the step waits for (cost model, to be tested in H1/E3). At 76 slots each rank sees
~660 unique local expert requests per step, so the LRU hit rate is ~88%.

### 2026-10-25 E1b: rank imbalance is noise, not placement (H4 as stated: refuted)

At 76 slots: mean misses rank0 84.2, rank1 76.2. Sum over layers of max(mean) = 86.4, but sum of E[max] =
111.0. Systematic per-layer imbalance explains ~2 misses; the other ~25 come from per-step fluctuation. The ranks
are positively correlated (mean per-layer r = 0.46). With independent ranks of equal distribution, E[max] would be
123.5. Re-placing experts across ranks cannot remove step-level noise, so H4's ≥10% via placement is refuted.
New hypothesis H6: pooling both PCIe links (the less-loaded rank fetches part of the busier rank's misses and
forwards them over NVLink) brings the effective count toward the per-rank mean (~80 at 76 slots).

### 2026-10-25 E1c: online policies (offline replay, 76 slots, same traces)

| policy | max-rank misses |
|---|---:|
| LRU (runtime) | 111.0 |
| LRFU λ=0.5 / 0.8 / 0.9 / **0.95** / 0.98 / 0.99 | 110.3 / 108.6 / 105.9 / **103.9** / 106.8 / 112.7 |
| LRU-2 | 109.3 |
| LRU with 16 / 32 / 48 seeded experts pinned | 118.0 / 129.2 / 149.3 |
| TinyLFU-style admission with bypass (λ=0.9–0.95) | 105.6–105.9 |
| admit only if requested in the last 1–16 calls | 119.7–134.5 |
| Belady with bypass (bound) | 57.9 |

The best online policy saves 6.4%. λ was tuned on the evaluation traces, so it still needs a held-out check. The
2× gap to Belady is future knowledge that history-based policies cannot recover. Pinning a static core is
harmful: demand shifts within a sequence (consistent with the hillclimb's LRU result).

### 2026-10-25 E2: prefill transient composition (H5 mechanism refuted)

`runs/e0b-hooks-on/memsnap-dev0-step34.pickle`, the 2nd 8,192-token chunk of a 131K prompt, F5. Allocator
baseline 12.6 GiB. Transient peak +1,166 MiB, with 117 MiB of driver-free memory left at the step's peak
(`stepmem`). The 51 allocations alive at the peak:

| MiB | site |
|---:|---|
| 480 | `rms_norm` (3 × 160 MiB = 8,192 tokens × 4 HC streams × 2,560 × BF16) |
| 160 + 160 | `hc_combine_norm`, `hc_combine` (hyperconnection) |
| 160 + 40 | PLE forward output, `_dequantize_embeddings` |
| 80 | PLE short conv |
| 45 | low-latency GEMM outputs |
| 40 | NCCL all-reduce buffer |

None of the peak is MoE-internal (stream staging slots live in borrowed VMM pages, outside the allocator). H5's
proposed mechanism, serving MoE temporaries from borrowed hot pages, would not lower this peak. The demand is the
residual-stream-sized HC/norm/PLE activations, which scale with chunk tokens (4 × 2,560 × 2 B = 20 KiB per token
per live copy). Options: (a) lend whole hot-page layers to the allocator during staged prefill and re-create them
afterwards; (b) cut live copies in the HC/norm path (in-place norm, fusing), ~0.3–0.5 GiB.

### 2026-10-25 E3: decode is miss-bound, 177 µs per max-rank miss (H1 supported)

`runs/e3-hot{64,72,76,80,88,96}`, F5 bundle with `MAX_MODEL_LEN=135168`, `KV_CACHE_MEMORY_BYTES=2500000000`,
4,096-token chunks, hooks on (per-step GPU timing via CUDA events at each ≤16-token `execute_model`). Decode
suite ×2: repo-chat 128+2048, chat code 1536, chat explain 1536. Order 76, 96, 64, 88, 72, 80. Each run regresses
the GPU time of every 4-token→4-token step on the runtime's own per-step miss counts (`analyze_steps.py`; slowest
1% of steps dropped):

| slots | steps | max-rank misses/step | mean step ms | fit: ms = a + b·max-rank | R² | decode tok/s (recip. mean, range) | acceptance |
|---:|---:|---:|---:|---|---:|---:|---:|
| 64 | 4,182 | 124.9 | 39.10 | 16.46 + 0.1806 | 0.999 | 62.3 (54.4–73.4) | 0.484 |
| 72 | 4,365 | 108.8 | 36.51 | 17.04 + 0.1782 | 0.999 | 63.9 (59.3–69.6) | 0.448 |
| 76 | 4,368 | 99.1 | 34.77 | 17.08 + 0.1778 | 0.999 | 67.3 (63.2–70.6) | 0.446 |
| 80 | 4,319 | 95.8 | 34.21 | 17.15 + 0.1771 | 0.998 | 69.1 (63.3–73.0) | 0.458 |
| 88 | 4,234 | 83.5 | 32.08 | 17.29 + 0.1760 | 0.998 | 75.1 (66.6–82.5) | 0.476 |
| 96 | 4,214 | 67.5 | 29.33 | 17.52 + 0.1735 | 0.997 | 82.9 (75.3–91.0) | 0.482 |

Within hot76: residual SD 0.31 ms over steps of 21–70 ms. The max-rank count explains step time better than
total misses (R² 0.993) or rank-0 misses (0.974), confirming the per-layer EP synchronization cost model. 177 µs
is one 2.42 MiB expert at 14.4 GB/s: the copy is fully serialized on the critical path (the M4 overlap hides
GEMM time, not copy time). Everything else in a verify cycle (target, MTP draft, sampling, PLE, collectives) is
a constant ~17 ms. At hot76, 51% of decode time is PCIe copies. Zero misses would give ~2× decode at the same
acceptance.

Consequences, all in ms per step at hot76 (99 max-rank misses): each slot saves ~0.2 ms (the hot72→hot96 slope);
LRFU (−6.4% misses) ~1.1 ms; link pooling to the per-rank mean (~71) ~5 ms; any mechanism that takes misses off
the PCIe critical path scales with the remaining 17.5 ms.

### 2026-10-25 E4: CPU compute of misses on this host

Host memory: 4 × 32 GB DDR5 at 3,600 MT/s (four DIMMs), dual channel; 7950X3D, two CCDs (0–7/16–23, 8–15/24–31).
`e4_cpu_bench.py` replays the hot76 LRU misses of the E1 traces (400 steps after 100 skipped; 86.7 / 80.6 misses
per step per rank, max-rank 115.2) through `cpu_moe.layer_forward`, one process per rank pinned to one CCD,
from the checkpoint's GPTQ tensors in the page cache, with a barrier after every layer. Per-step sum over layers
of the per-layer max-rank CPU latency:

| variant | CPU ms/step | PCIe model (0.176 ms × max-rank) |
|---|---:|---:|
| 7 threads/CCD, 1 slice (default) | 15.12 | 20.28 |
| 14 threads/CCD (SMT), 1 slice | 16.79 | 20.28 |
| 7 threads/CCD, 3 slices | 11.25 | 20.28 |
| **7 threads/CCD, 5 slices** | **10.84** | 20.28 |

`e4b_micro.py` (one process, distinct random experts, so no L3 reuse; medians): an empty call costs 0.8 µs. One
expert on one thread costs 235 µs (10.8 GB/s, the single-core DRAM read limit). The default kernel gives one
expert to one thread, so a decode layer with 1–3 misses leaves most threads idle and costs ~240–270 µs. With
`set_slices` at 5 × 128 columns and 7 threads: 1/2/3/4/8 experts take 78/144/202/257/504 µs, ~63 µs per
expert at saturation (40 GB/s from one CCD). Both ranks at once in the replay: ~95–100 µs per expert. The CPU's
marginal cost per expert is ~1.9× lower than a PCIe copy (176 µs), and a whole step's misses take 47% less time.

Offline hybrid model (`lru_sim.simulate_split`, E1 traces, hot76; per layer, the slower rank's max of PCIe-part
× 176 µs and CPU-part × 95 µs):

| miss handling | misses/step r0 / r1 | miss-path ms/step |
|---|---:|---:|
| all PCIe copy + insert (runtime) | 84.2 / 76.3 | 19.55 |
| all CPU, nothing inserted | 293.4 / 260.2 | 32.43 |
| 25% PCIe + insert, 75% CPU | 100.6 / 89.9 | **10.25** |
| 25% PCIe + insert, CPU ones also inserted after the step | 84.2 / 76.3 | 9.21 |
| 35% / 50% PCIe + insert, rest CPU | 95.3 / 93.7 (r0) | 11.69 / 12.72 |

Inserting some misses is essential: without it the cache stops adapting. Predicted hybrid at hot76:
17.2 + 10.3 ms + bridge overhead, versus 34.8 ms measured, before any CPU/GPU overlap within a layer.

Not yet known: the per-layer cost of the GPU↔CPU bridge in the real runtime, and DRAM contention with the PCIe
copies and the rest of the serving process.

### 2026-10-25 E5 (prediction written before the run): the 64 GB hot-only profile on this host

`configs/2x3090-64gb.env` (static hot88 per rank, CPU pool per CCD, GPU share 0.3 via UVA GEMM, NVMe PLE mmap),
decode suite ×2 as in E3; s1 = default slicing, s5 = `QWEN38_CPU_EXPERTS_SLICES=0,128,256,384,512,640`.
Prediction from the E3-hot76 traces (same prompts): 230.0 / 195.1 cold experts per step per rank. Step time =
17.2 ms + miss path + bridge overhead; miss path 18.95 ms at 95 µs per CPU expert (s5), ~27 ms at ~160 µs
effective (s1). So s5 should take ~36 ms + bridge per step and s1 ~44 ms + bridge. The measured difference from
these numbers is the in-situ bridge overhead per step.

### 2026-10-25 E5 result: in-situ CPU path on this host (s1 prediction held, s5 refuted)

| run | mean step ms (4→4 steps, ≤p99) | median | decode tok/s range | acceptance |
|---|---:|---:|---:|---:|
| E5-s1 64 GB profile, default slicing | 44.25 | 42.30 | 45.4–55.1 | 0.37–0.49 |
| E5-s5 same, 5 slices | 45.05 | 42.49 | 51.0–58.8 | 0.41–0.52 |
| E3 hot76 LRU, PCIe only | 34.26 | 32.37 | 63.2–70.6 | 0.41–0.48 |
| E3 hot88 LRU, PCIe only | 31.62 | 29.71 | 66.6–82.5 | 0.40–0.58 |

s1 matched its prediction (~44 ms), so the in-situ bridge overhead is small. The s5 prediction (~36 ms) was refuted:
the linear 95 µs-per-CPU-expert model does not hold at 3–4 CPU experts per layer and rank, where E4b shows slicing
gains little (288 vs 257 µs for 4 experts). Slicing helps at 1–2 experts per layer, which is the LRU-hybrid
regime, not this one. The static hot-only design is 29% slower per step than PCIe LRU at hot76 on this host: its
cold set per step is 2.3× the LRU's miss count, more than the cheaper CPU path can make up.

Open: an LRU + CPU split (simulated miss path 10.3 vs 19.6 ms at hot76) needs a CPU kernel that reads the experts'
Humming layout, because RAM cannot hold a second GPTQ copy of the 60 GB host pool. Its benefit rests on the
1–2-expert-per-layer CPU latency (78–144 µs isolated, ~97 µs per expert with both ranks), which E4/E4b measured
outside the runtime.

### 2026-10-25 E1d: per-layer slot allocation (H3 partly supported)

`h3_allocation.py`: greedy allocation of the same total (48 × 76 = 3,648 slots per GPU, both GPUs alike) in steps
of 4 slots by marginal miss reduction on the fit traces' per-layer LRU curves (mean of both ranks), then exact
replay on the other trace set with the max-rank metric. Constraints: 128 slots max per layer; floor 64 or 32; the
last 8 layers keep ≥64 each so stream staging can still borrow 512 rows.

| fit → eval | floor | uniform max-rank | fitted max-rank | change |
|---|---:|---:|---:|---:|
| E1 traces → E3-hot76 traces | 64 | 98.48 | 90.62 | −8.0% |
| E3-hot76 → E1 | 64 | 111.00 | 104.53 | −5.8% |
| E1 → E3-hot76 | 32 | 98.48 | 89.19 | **−9.4%** |
| E3-hot76 → E1 | 32 | 111.00 | 104.10 | **−6.2%** |

The fitted allocations are stable in shape: layers 0–2 get 112–128 slots, layers 15/16 and 31/32 get 44–52.
Per-layer miss rates at uniform hot76 range from 0.53 (layer 31) to 5.26 (layer 0) per step and rank, about 10×.
H3's bar (≥10%) is not met, but −6 to −9% at zero VRAM equals roughly 4–6 extra uniform slots. Caveat: the two
trace sets share three prompts (repo-chat 128, chat code, chat explain), so this is not a fully independent
validation.

### 2026-10-25 E1c addendum: LRFU on the E3 trace set

E3-hot76 traces (not used to choose λ, but sharing three prompts with E1): LRU 98.48, LRFU λ=0.9 94.86, **λ=0.95
94.09 (−4.5%)**, λ=0.98 97.58 max-rank misses per step. The same λ is best on both trace sets. The combination of
LRFU with the per-layer allocation has not been simulated yet.

### 2026-10-25 E6 (prediction written before the run): hot-set frontier at 128K context with 8K chunks

Design: F5 bundle with `MAX_MODEL_LEN=135168`, `KV_CACHE_MEMORY_BYTES=2500000000` (the agent-128k pool) and
8,192-token chunks; hot 76 (control: same context, isolates the hot-set effect), 88, 92, 96, 100, run in the order
92, 76, 100, 88, 96. One hooked session per point (E0 found the hooks inert; this deviates from the plan's
hooks-off prefill runs to halve the startups): `decode` ×2 (step time, misses), then `long` ×2 (prefill, TTFT,
post-prefill decode, VRAM peak from STEPMEM, preemptions).

Prediction. Memory: the smaller pool frees 1.93 GB per GPU, 16 slots at 121 MB, so hot92 has F5's headroom.
agent-128k-prefill (hot88, 8K, no EMBED_UVA) peaked at 23.8–24.0 GiB; EMBED_UVA frees 0.6 GB (5 slots). So 88 and
92 run, 96 is marginal (likely OOM in the first 8K chunk of the 131K request), 100 fails. Speed: prefill
unchanged at ~4,000 tok/s for every hot set (staging copies all 256 rows per layer regardless). Short decode
from the E3 curve: hot76 ~67–71, hot88 ~75, hot92 ~79 tok/s; 131K post-prefill decode up by the same ratio.

### 2026-10-25 E6 result: hot96 fits at 128K with 8K chunks; decode +19–22% over hot76, prefill unchanged

Runs `runs/e6-hot{76,88,92,96,100}` (order 92, 76, 100, 88, 96), one session each: `decode` ×2 (six requests)
and `long` ×2 (131,072+2,048). The hooks' LRU and step-timing dumps were lost: the E5 edit to `_dump_lru` dropped
its `for dev, rec in _recorders.items():` line, so nothing was written (fixed after E6; the E5 step files were
unaffected). Decode below is API-observed (reciprocal mean TPOT), not per-step GPU time.

| hot | short decode (6 req) | 131K post-prefill decode (2) | 131K prefill tok/s (TTFT s) | driver-free at 8K peak, MiB (GPU0, GPU1) |
|---:|---:|---:|---|---|
| 76 | 67.66 | 69.79 | 3,934 / 4,119 (33.3 / 31.8) | 1,043 / 1,057 |
| 88 | 76.76 | 80.99 | 4,046 / 4,292 (32.4 / 30.5) | 673 / 687 |
| 92 | 77.36 | 81.55 | 4,107 / 4,308 (31.9 / 30.4) | 287 / 301 |
| 96 | 80.84 | 85.18 | 4,067 / 4,307 (32.2 / 30.4) | 91 / 105 |
| 100 | (79.6–89.0 per request) | OOM | first 131K request: `torch.OutOfMemoryError` (48 MiB) in the GDN chunk kernel, 5th 8K chunk | – |

MTP acceptance 0.45–0.46 at every point; zero preemptions; one 133K request is 98.6% of the 135,168 window.

Interpretation:
- The memory prediction was one step conservative: 96 fits (not just 92), with 91–105 MiB driver-free at the
  worst 8K step (F5 at 256K had 117). 100 fails in the GDN chunk peak, the one E2 ranked second. Driver-free does not
  fall linearly with slots (76→88 cost 370 MiB, not 1.45 GB) because the caching allocator's reserve differs between
  sessions (hot76 reserved 14.4 GB, the others 13.0 GB); treat it as headroom, not as a slot ledger.
- Decode: hot96 vs hot76 at the same context is +19.5% short and +22% long, in line with E3's step law (predicted
  −16% step time ⇒ +19% tok/s). hot88→92 gained little (+0.8%), within request-to-request noise (single requests
  span 62–93 tok/s; acceptance varies per request).
- Prefill is unchanged across hot sets (3,934–4,308 tok/s, second request ~5% faster as in F5), as predicted.
- Practical frontier at 128K: hot96, 8K chunks: ~4,300 tok/s prefill and ~85 tok/s decode after 131K, versus F5
  (256K) ~4,100 and ~74. The cost is half the context and 91 MiB of headroom; a hot92 profile keeps ~290 MiB.

### 2026-10-25 E7 (prediction written before the run): exact prefill-memory trims

Candidate (`e7/model.py`, `e7/ple_layer.py`, mounted with `OVERLAY_OVERRIDES`; no op changes): the PLE forward drops
`embeddings` after the projections, the normalized `key`/`query` right after the gate and `value` after the
gating; the decoder loop passes the stream state to each layer in a one-element list that the layer empties, so
the previous layer's state dies at the layer's first combine instead of living through the whole layer.
Design: F5 (`.env`, 256K, hot76) with step-memory hooks, control then candidate; each runs the `exact` probe twice
(greedy, prompt logprobs at every position plus 32 output logprobs, prompts of 128 and 20,000 tokens and one chat)
and one 131K+2K request.
Prediction: candidate logprobs and tokens identical to the control wherever the control agrees with itself;
peak allocated per 8K step −260 MiB (PLE peak −480 → no longer binding; GDN and HC peaks −160); prefill and decode
unchanged within noise. E6's hot100 failure was at the GDN peak and needs ~380 MiB more than hot96, so E7 alone
should not admit hot100 at 128K; at 256K it should admit hot78 (2 slots), not hot80.

### 2026-10-25 E7 result: trims −258 MiB per 8K step (prediction held); exact by construction, equal within noise end to end

Runs `runs/e7-ctl`, `runs/e7-trim` (F5, 256K, hot76; step-memory hooks only). The first attempt crashed the
engine: prompt logprobs on a 20K prompt need full-vocabulary FP32 logits for an 8K chunk (~8 GB); the probe now
asks for prompt logprobs only up to 2,048 tokens (the 20K prompt keeps its 32 output logprobs).

| | control | trims |
|---|---:|---:|
| transient per 8K step (peak − after), MiB | 1,126 | 868 |
| peak allocated, MiB | 13,708 | 13,450 |
| min driver-free at an 8K step, MiB (GPU0 / GPU1) | 213 / 229 | 533 / 549 |
| 131K prefill tok/s (TTFT s) | 4,055 (32.3) | 4,135 (31.7) |
| post-prefill decode tok/s | 69.8 | 71.1 |

Exactness. The control is not deterministic: two control runs differ by up to 2.06 nats in prompt logprobs
(median 0.005), so a bitwise end-to-end test is impossible on this stack (approximate QSA top-k, split reductions).
Greedy tokens were identical in all 12 comparisons. |Δlogprob| distributions over the 321 positions per pair:
control vs control median 0.0051 / p95 0.81; candidate vs candidate 0.0044 / 0.53; control vs candidate 0.0076 /
0.77. The change only drops Python references (no op, dtype, shape or kernel argument changes; the caching
allocator's 512-byte alignment keeps kernel specializations), so it is exact by construction; end to end it is
indistinguishable from the control's own noise.

Interpretation: −258 MiB matches the −260 prediction (the PLE peak no longer binds; the GDN chunk peak does, minus
the stale residual). Headroom at 256K grows by ~320 MiB, about 2.6 slots: hot78 should fit, hot80 (+484 MB) is
marginal. At 128K, hot100 needs ~380 MiB more than hot96 had; the trims give ~260, so hot100 should still fail.

### 2026-10-25 E8 kernel check (offline, GPU, before any serving run)

`e8_kernel_test.py` feeds every recorded call (E0 traces, layers 0/17/40, both ranks, 44,502 calls, hot76) through
the runtime `update_kernel` with the `e8/` files mounted. λ unset: 100.000% of calls give the simulator's miss
count (the LRU path is unchanged). λ = 0.95: 99.973% (12 calls), equal miss totals within 3 of 35,000: ties and
float32 keys versus the simulator's float64 scores. The LRFU keeps one key per expert after the slot keys in
`slot_ages` (512 extra int32), so a score survives eviction; the forgetting variant was simulated first and kept
less of the gain (E3 traces: −1.7% at λ 0.95, −3.3% at its best λ 0.9, versus −4.5% with history).

### 2026-10-25 E7b (prediction): largest hot set at 256K with the trims

F5 + trims at hot80 (fallback hot78), suites: 262,016+128 first, then 131K+2K, then `decode` ×1. Prediction: hot80
completes 262,016+128 with < 100 MiB driver-free at the worst 8K step (533 − 4 × 121 MB ≈ 70 MiB) or fails in
the GDN chunk; hot78 fits. Decode at hot80 vs hot76 (E3 law): ~+4%.

### 2026-10-25 E8 (prediction written before the run): LRFU and per-layer allocation in the runtime

Design: F5 (256K, 3,648 slots per GPU = 48 × 76) with the `e8/` MoE files mounted in every session; `heldout` ×2
(six new chat prompts, 1,536 output tokens each, domains and wording absent from the E1/E3 traces the policies were
fitted on). Sessions in order: released LRU (λ and sizes unset), LRFU λ = 0.95, per-layer allocation (fitted with
`H3_FLOOR=32` on the E0 traces: layer 0 128 slots ... layers 15/31 44; last 8 layers 520 rows), both, released
LRU again (drift). Metrics: max-rank misses per 4-token step from the recorder and per-step GPU time (H1 fit),
plus API decode and acceptance.
Prediction: from the offline replays (−4.5 to −6.4% LRFU, −6.2 to −9.4% allocation), max-rank misses per step
fall ~5% (LRFU), ~8% (allocation) and ~10–13% (both, if roughly additive). With step = 17.2 + 0.177 × misses at
~100 misses, that is ~2.5%, ~4% and ~5–6% shorter verify steps. API decode gains of that size are below the
request-to-request noise; the per-step GPU time and miss counts are the deciding metrics. Offline replay of the
control's held-out traces must predict each candidate's measured misses within ~2%.

### 2026-10-25 E7b result: hot80 fits at 256K with the trims (full-context probe passed)

`runs/e7b-hot80` (F5 + trims + hot80; one session): 262,016+128 first request: TTFT 65.9 s (3,977 tok/s), zero
preemptions; 131K+2K: 4,192 tok/s (31.3 s), 72.8 tok/s decode; `decode` ×1: 66.9–75.7 tok/s. Worst 8K step
driver-free 107 / 121 MiB (48 large steps; F5 hot76 without trims: 117). The prediction (~70 MiB, marginal) was
pessimistic by ~40 MiB; hot78 was not needed. So at 256K the trims buy 4 slots at F5's original margin. Decode gain
over hot76 is not resolved by one session (E3 law: ~+4%).

### 2026-10-25 E8 result: LRFU and per-layer allocation add up: −15% misses, −7.5% verify step, +6% decode (held-out)

Runs `runs/e8-{lru,lrfu,alloc,both,lru2}` (F5, 256K, 3,648 slots/GPU; `heldout` ×2 = 12 requests each). Two
operational failures on the way, both runner bugs, not policy effects: (1) the allocation session crashed at
startup because `stream_stage.py` stacked every layer's `slot_global_ids` (equal lengths assumed); `e8/stream_stage.py`
concatenates and splits instead (mounted from then on; for equal sizes it is the same data, so the earlier lru and
lrfu sessions are comparable). `session.sh` did not notice the dead engine and sat in the client's 30-min wait; it
now fails on `Engine core initialization failed` or after 25 min. (2) A 10-min startup cap I added then killed the
first `both` session during a slow weight load (`runs/e8-both-slowstart`); rerun with 25 min.

| session | 4-token steps | max-rank misses/step | verify step ms | step fit | API decode (12 req) | acceptance |
|---|---:|---:|---:|---|---:|---:|
| released LRU | 7,069 | 105.11 | 35.29 | 16.98 + 178.5 µs | 72.79 | 0.550 |
| LRFU λ 0.95 | 7,222 | 97.24 (−7.5%) | 34.09 (−3.4%) | 17.27 + 177.7 µs | 73.73 | 0.530 |
| per-layer allocation | 7,170 | 96.36 (−8.3%) | 33.66 (−4.6%) | 17.07 + 177.0 µs | 75.13 | 0.535 |
| both | 7,155 | 89.36 (−15.0%) | 32.64 (−7.5%) | 17.34 + 176.3 µs | 77.63 | 0.537 |
| released LRU again | 7,195 | 103.20 (−1.8%) | 34.94 (−1.0%) | 17.01 + 178.3 µs | 72.18 | 0.534 |

Against the mean of the two LRU sessions (drift 1.8% misses, 1.0% step): LRFU −6.6% / −2.9%, allocation −7.5% /
−4.2%, both −14.2% misses / −7.1% step, API decode +7.0% (77.63 vs 72.49).

Offline replay of the first LRU session's held-out traces predicted −5.6% (LRFU), −8.3% (allocation) and −12.9%
(both); measured −6.6 to −7.5, −7.5 to −8.3 and −14.2 to −15.0. All within ~2 points, so the simulator ranks
runtime policies correctly on unseen prompts, and the two gains are nearly additive (as predicted, slightly better).

Verdict: H3 (layers differ) supported in the runtime; the online-policy gain is real and transfers to held-out
prompts. The step law is unchanged (17.0–17.3 ms + 176–179 µs per max-rank miss): the policies change only the
miss count. Remaining caveats: one held-out set (six chats, short context), the λ and allocation were not re-tuned
for each other, and long-context decode (131K) was not measured with the new policies.

### 2026-10-25 E9 (prediction written before the run): combined candidates vs F5

Design (`e9.sh`), one day, order F5, 256K candidate, 128K candidate, F5 again (drift). Hooks: LRU recorder, step
timing, step memory. Candidate runtime = E7 trims + E8 files, LRFU λ 0.95 (`overrides/e9-candidate.env`).
- 256K candidate: F5 + hot80 average, per-layer sizes refitted at 80 (`e9-alloc80.env`, fit on E0 traces:
  −8.7% on the E8 held-out traces). Suites: `full` (262,016+128), `long` ×2, `decode` ×1, `heldout` ×1.
- 128K candidate: 135,168 context, 2.5 GB KV, 8K chunks, 96 average, sizes refitted at 96 (`e9-alloc96.env`,
  −10.1% held-out). Suites: `long` ×2, `decode` ×1, `heldout` ×1.

Prediction (`e9_predict.py`: replay of the E8 control's held-out traces, then the E3 step law):
max-rank misses F5 104.9, 256K candidate 84.4 (−19.5%), 128K candidate 62.6 (−40.3%); verify step 35.8 → 32.1
(−10%) and 28.3 ms (−21%). Measured on the `heldout` and `decode` suites, which differ from the replayed traces,
so expect agreement within ~3 points, not exactly. Prefill unchanged within noise (~4,000–4,300 tok/s at 131K);
the 256K candidate passes 262,016+128 with ~100 MiB driver-free at the worst 8K step; the 128K candidate keeps
≥250 MiB (E6 hot96 91 MiB + E7 trims ~260). Post-131K decode: +~10% (256K) and +~25% (128K) over F5.

### 2026-10-25 E9 result: both candidates OOM'd; the margins were thinner than single runs showed

`runs/e9-{f5,256k,128k,f5b}`. Both F5 sessions completed everything (262,016+128: 3,930 / 3,951 tok/s; 131K:
4,104–4,131 tok/s, 68–72 tok/s decode). Both candidates failed in their first long request:
- 256K (trims + hot80 + LRFU + sizes@80): `torch.OutOfMemoryError` on a 320 MiB request (HC mix grouped RMSNorm
  output of layer 0 at an 8K chunk) in the 3rd 8K chunk of 262,016+128, with 95 MiB free and **786 MiB reserved
  but unallocated**: fragmentation. E7b passed the same request at hot80 with uniform sizes; hot80 at 256K is
  therefore not robust (one pass, one fail at ~100 MiB nominal margin). E7b's conclusion is revised.
- 128K (trims + hot96 + LRFU + sizes@96): OOM on the same 320 MiB request inside the MTP draft's prefill, 3rd 8K
  chunk of the 131K request; 3 MiB driver-free already at the first 8K step, 674 MiB reserved-unallocated.

Diagnosis (first 8K step; non-PyTorch = total − driver-free − reserved):

| run | reserved | peak allocated | driver-free | non-PyTorch |
|---|---:|---:|---:|---:|
| e6-hot96 (no trims, uniform) | 12,492 | 11,989 | 91 | 11,586 |
| e9-128k | 12,412 | 11,689 | 3 | 11,754 (+168) |
| e7b-hot80 (trims, uniform) | 14,216 | 13,455 | 107 | 9,846 |
| e9-256k | 14,036 | 13,480 | 237 | 9,896 (+50) |

- Per-layer sizes cost 50–170 MiB outside PyTorch: every session with them captured graphs in 27–44 s using
  0.21–0.22 GiB (vs 5 s / 0.10 GiB), and started 7–9 min slower (16–17 distinct capacities, each its own LRU
  warmup and Humming `num_experts = 256 + capacity` configuration). The LRU update kernels themselves are only
  22–28 MiB for 136 specializations (`e9_module_mem.py`); LRFU adds ~6 MiB.
- The trims lower peak allocation by ~300 MiB, but reserved memory falls only ~80–180 MiB: the freed blocks stay
  in the caching allocator as fragments, and the binding event is a single 320 MiB request.
- E8 never saw this because its held-out prompts are short (no 8K prefill).

Lesson for the protocol: a configuration is only "fits" after ≥2 passes of the full-context (or full-window)
request, and candidates must keep at least the control's margin (F5: 117–193 MiB driver-free at the worst step).

### 2026-10-25 E9c (prediction): margin-preserving candidates

- 256K: F5 hot76 + trims + LRFU + sizes@76 (the E8 allocation). Memory vs F5: trims +~180 reserved, sizes −50.
  Prediction: passes 262,016+128 with ≥ F5's margin; held-out misses −14%, verify step −7% (E8), prefill unchanged.
- 128K: hot92 + trims + LRFU + sizes@92 (refit). E6 hot92 had 287 MiB free; −170 sizes, +~80–180 trims.
  Prediction: completes 131K ×2 with ≥150 MiB free; max-rank misses ~−35% vs F5, verify step ~−18%.

### 2026-10-25 E9c result: both margin-preserving candidates pass and win (controls: e9-f5, e9-f5b)

`runs/e9c-256k` (F5 hot76 + trims + LRFU + sizes@76), `runs/e9c-128k` (128K, hot92 average + trims + LRFU +
sizes@92). Every request completed, zero preemptions. Per-step numbers are 4-token verify steps from the recorder,
grouped by workload window (F5 = mean of its two sessions).

| | F5 | 256K candidate | 128K candidate |
|---|---:|---:|---:|
| held-out: max-rank misses / step | 103.3 | 91.1 (−11.8%) | 65.1 (−37.0%) |
| held-out: verify step ms | 35.05 | 32.99 (−5.9%) | 28.54 (−18.6%) |
| after 131K: max-rank misses / step | 113.1 | 98.8 (−12.7%) | 68.1 (−39.8%) |
| after 131K: verify step ms | 37.22 | 34.89 (−6.3%) | 29.63 (−20.4%) |
| API decode, held-out (6 req per session) | 72.53 | 76.78 (+5.9%) | 87.74 (+21.0%) |
| API decode, `decode` suite (3 req) | 69.71 | 73.28 (+5.1%) | 81.07 (+16.3%) |
| API decode after 131K (2–3 req) | 69.90 | 75.82 (+8.5%) | 88.65 (+26.8%) |
| 131K prefill tok/s | 4,104–4,131 | 4,148 / 4,169 | 3,951 / 4,309 / 4,279 |
| 262,016+128 prefill tok/s | 3,930 / 3,951 | 3,940 / 4,126 | (128K window) |
| min driver-free at an 8K step, MiB | 193–207 | 321–335 | 281–295 |

Predictions: 256K −14% misses / −7% step → measured −12 to −13% / −6%; 128K −35% / −18% → −37 to −40% / −19 to
−20%. Both inside ~2 points. Prefill unchanged. Both candidates keep more headroom than F5 itself.

Two first-use stalls, both compiles caused by new slot counts, not steady-state costs:
- 256K, first 262,016+128: one 11.8 s gap during decode = `update_kernel` JIT. `lru_warmup.py` always warmed
  with `capacity_block=128`, but the runtime passes `next_power_of_2(capacity)`, which is 64 for layers with
  ≤64 slots. Uniform sizes ≥65 never hit it; per-layer sizes down to 44 do. Fixed in `e8/lru_warmup.py` (warm
  the runtime's own specialization); not yet measured.
- 128K, first 128-token request after the long ones: TTFT 20.7 s = 63 Humming kernels built for the short-prefill
  shapes of each new capacity (`num_experts = 256 + capacity`), written to `jit-cache/humming` at that moment.
  Persistent cache: a second start would not pay it. Each distinct capacity also loads its own modules (the
  50–170 MiB non-PyTorch cost found in E9).

Verdict: the combined runtime (trims + LRFU + per-layer sizes) is a real, robust gain at both context sizes.
At 256K it buys ~6% decode at a larger memory margin than F5; at 128K with hot92 it gives ~+21–27% decode with
unchanged ~4,100–4,300 prefill. Caveats: one session per candidate; per-request API decode still varies ±10%
with MTP acceptance; first-start compiles until the JIT cache is warm.

### 2026-10-25 Promotion: overlay, configs, release validation

User decision: promote the 256K candidate; the 128K candidate as an optional config. Overlay files replaced
(`model.py`, `ple_layer.py`, `compressed_tensors_moe_wna16.py`, `lru_map.py`, `stream_stage.py`, `lru_warmup.py`;
the six repo files were first checked byte-identical to the measured image), `SHA256SUMS.json` updated,
`configs/fast-256k-prefill.env` gains the two policy lines, new `configs/fast-128k-prefill.env`, image rebuilt as
`:locked` (previous image kept as `:locked-2026-10-03`). Two changes beyond what E9c measured, both re-checked with
`e8_kernel_test.py` (LRU 100.000%; LRFU 99.95–99.99% at 44 and 76 slots):
- LRFU key rebase every 65,536 calls (float32 resolution on long-running servers); forced at 1,000 calls in the
  test: 99.987%.
- `capacity` is a runtime argument of both LRU kernels. The first validation start with it as a constexpr was
  still warming up after 25 min (`runs/v-256k-slowstart`): every distinct size × 12 counts compiled, plus the
  warmup fix's 64-wide variants. Now 15 warmups took 470 s on an empty cache, two of them carrying nearly all of
  it (the LRFU serial fallback unrolls up to 120 ids), the rest ~1 ms.
Operational: I launched the second validation while the first `v.sh` was still running its later sessions; the
two raced for the container name and GPUs. Stop the old driver (and its `session.sh` children) before relaunching.

Release validation (`runs/v-{256k,f5,128k}`, hooks off, one session each; published as `benchmarks/2026-10-25/`):

| | control (trims on, policies off) | 256K profile | 128K profile |
|---|---:|---:|---:|
| decode held-out (6) | 74.03 | 78.57 (+6.1%) | 89.78 (+21.3%) |
| decode `decode` suite (3) | 68.20 | 74.57 (+9.3%) | 86.47 (+26.8%) |
| decode after 131K (2) | 73.29 | 80.45 (+9.8%) | 86.95 (+18.6%) |
| 131K prefill tok/s | 4,182 / 4,178 | 4,198 / 4,200 | 3,981 / 4,342 |
| 262,016+128 | 3,978 / 4,145 | 3,968 / 4,157 | — |

No LRU-kernel compile during inference (the warmup fix holds). First short request TTFT 4.8–5.7 s in the policy
sessions (control 1.9 s): remaining first-use Humming compiles. Production `.env` now carries the two policy lines
(backup in the session scratchpad); production restarted on the new image.
