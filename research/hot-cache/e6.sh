#!/usr/bin/env bash
# E6: hot-set frontier at 128K context with 8,192-token chunks, interleaved order. One hooked session per point
# (E0: hooks are inert): decode x2 for step time and misses, then long x2 for prefill, TTFT and post-prefill decode.
set -uo pipefail
cd "$(dirname "$0")"
for n in ${E6_POINTS:-92 76 100 88 96}; do
  ./session.sh runs/e6-hot$n "decode:0:2 long:0:2" overrides/e6-hot$n.env || echo "hot$n failed"
done
