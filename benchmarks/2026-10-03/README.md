# PLE table on a third GPU and the x8 prefill profile — October 3

Two results on one host, measured with the same locally built image (this tree's `docker/Dockerfile`,
image ID in [`summary.json`](summary.json); not a published image):

1. `QWEN38_PLE_GPU` ([`configs/ple-gpu.env`](../../configs/ple-gpu.env)) moves the PLE n-gram table to a
   third GPU that is not a TP rank. Speed is unchanged; the host keeps about 13 GiB more RAM.
2. [`configs/fast-256k-prefill.env`](../../configs/fast-256k-prefill.env) uses that headroom for 8,192-token
   prefill chunks at 256K: 131K prefill went from about 865 to about 4,000 input tok/s with decode unchanged.

## Host

| | |
|---|---|
| GPUs | 2× RTX 3090 24 GB on CPU root ports, PCIe 4.0 ×8 each (AM5 lane split), NVLink bridge, CUDA P2P OK; RTX 5060 Ti 16 GB on a chipset slot, PCIe 3.0 ×4, no P2P to the 3090s |
| CPU / RAM | AMD Ryzen 9 7950X3D, 124 GiB; swap 3.5 GB for the first two rows below, 64 GB + 3.5 GB afterwards |
| Measured links | 3090: 13.4 / 13.2 GB/s pinned H2D / D2H, 7.5 µs per 4 KiB copy. 5060 Ti: 2.8 / 2.9 GB/s, 15.5 µs |
| Client | `scripts/benchmark_serving.py`, `repo-chat`, greedy, unique cache salt per request |

Shapes: 128 + 256, ten measured runs after one warmup (reciprocal-mean decode); 131,072 + 2,048, two runs,
no warmup; 262,016 + 128 once on the final profile. Raw client reports and 10-second host-memory traces are
in [`raw/`](raw/).

## Results

| Configuration | 131K prefill, input tok/s | 131K TTFT | Decode 128+256 | Decode after 131K | Host RAM available | Swap-in during 131K runs | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Checked-in `.env` (hot84, 2,048-token chunks), PLE worker on CPU | 844 / 936 | 155.2 / 140.0 s | 67.26 | 69.76 | 6.3 GiB | 129 KiB/s mean, 4.0 MB/s peak | 23,140 MiB |
| Same `.env`, PLE table on the 5060 Ti | 867 / 864 | 151.1 / 151.8 s | 70.61 | 66.98 | 19.0 GiB | 2 KiB/s | 23,140 MiB |
| F1: fast-256k bundle, 4,096-token chunks, hot84 | 2,285 / 2,335 | 57.4 / 56.1 s | 76.17 | 81.34 | 21.2 GiB | 243 KiB/s | 23,950 MiB |
| F2: fast-256k bundle, 8,192-token chunks, hot72 | 3,911 / 4,123 | 33.5 / 31.8 s | 68.47 | 71.99 | 21.4 GiB | 188 KiB/s | 23,834 MiB |
| F3: 8,192-token chunks, `QWEN38_EMBED_UVA=1`, hot80 | out of memory | | 72.73 | | | | |
| F4: 12,288-token chunks, `QWEN38_EMBED_UVA=1`, hot72 | out of memory | | 69.21 | | | | |
| **F5 (`configs/fast-256k-prefill.env`): 8,192-token chunks, `QWEN38_EMBED_UVA=1`, hot76** | **3,976 / 4,196** | **33.0 / 31.2 s** | **71.32** | **72.02** | 20.2 GiB | 9 KiB/s | 23,610 MiB |

F5 boundary probe, 262,016 + 128: 63.2 s to first token (4,147 input tok/s), 128 tokens generated, zero
preemptions, VRAM peak 23,610 MiB, host RAM available never below 20 GiB. The 128-token decode window is
capacity evidence only.

F3 and F4 failed with `torch.OutOfMemoryError` in the first prefill chunk of the 131K request (80 MiB
requested with 25 MiB free; 240 MiB requested with 117 MiB free). With `expandable_segments:False` the
usable headroom is smaller than the nvidia-smi peak suggests.

## What the numbers mean

- Prefill on this host is bound by expert streaming: every chunk copies all cold experts of a GPU (about 21 GB)
  over a PCIe 4.0 ×8 link, at least 1.55 s per chunk. 2,048-token chunks therefore cap prefill near 1,300
  tok/s; the chunk size is the lever, and VRAM limits it. 8,192-token chunks need hot72 at 256K; the input
  embedding in pinned host memory (`QWEN38_EMBED_UVA=1`, 1.27 GB of RAM) buys four experts back.
- The PLE table on the 5060 Ti does not change speed (the lookup is off the critical path when the table is
  resident). It converts 47.7 GiB of anonymous host RAM into 14.2 GiB of VRAM plus 33.5 GiB of pinned RAM,
  which removed the swap-in the baseline showed and made `QWEN38_EMBED_UVA` affordable.
- Twelve `QWEN38_PLE_GPU_VERIFY` lookups, including six full 2,048-token chunks, matched the checkpoint rows
  bit-exactly. Output-token hashes still differed between configurations, as greedy output does on this
  runtime (MTP batches, approximate QSA).

## Quality check: baseline versus F5

Same image; baseline is the checked-in `.env` with the CPU PLE worker and the 5060 Ti not exposed, F5 is the
new profile. Two kinds of evidence, each run twice per configuration so that run-to-run variation of the
runtime itself (MTP batches, approximate QSA) is visible ([`raw/quality-check.json`](raw/quality-check.json)).

Fixed public texts scored with `prompt_logprobs` (about 500–650 tokens each; longer texts run the cards out
of VRAM because every prompt token materializes full-vocabulary logits):

| Text | Tokens | Within-config \|Δ\| (baseline / F5) | Cross-config \|Δ\| (4 pairs) | F5 − baseline per token |
|---|---:|---:|---:|---:|
| docs/architecture.md | 514 | 0.88 / 2.37 nats | 0.75–3.99 | −0.0046 nats |
| docs/performance.md | 644 | 4.30 / 1.72 | 3.08–9.11 | +0.0095 |
| docs/memory.md | 528 | 9.02 / 0.96 | 0.13–10.11 | −0.0097 |
| docs/hardware.md | 570 | 6.46 / 4.38 | 1.72–8.19 | +0.0049 |

The per-token mean |Δ| across configurations (0.10–0.12 nats) equals the within-configuration value
(0.09–0.12), and the signed differences alternate in sign and average about zero. For scale, the repo rejected
INT8 dense weights at +0.010 nats per token.

Eight checkable greedy prompts (math, two code tasks executed against test cases, retrieval from the ~15K-token
AGENTS.md, a fact, JSON formatting, a tool call): 15 of 16 passes in both configurations with the same miss (the
arithmetic prompt with thinking off, where both configurations produced the same two wrong answers in the same
order). Answer text was byte-identical across configurations on every prompt where a configuration was
identical with itself; the three prompts whose text differed across configurations also differed between two
passes of the same configuration. Replies are not published.

## Caveats

Two long runs per configuration; differences under about 5% are noise. F2's first long run included JIT
compiles for the new chunk shapes; the persistent JIT cache was warm for F3–F5. MTP acceptance was not
recorded per run. These are measurements on one x8 host with the table on a 16 GB card; they do not change the
published 3090 claims, which come from an x16 host and different profiles.
