#!/usr/bin/env bash
# E3: decode vs hot-slot count, interleaved order.
set -uo pipefail
cd "$(dirname "$0")"
for n in 76 96 64 88 72 80; do
  ./session.sh runs/e3-hot$n "decode:0:2" overrides/e3-hot$n.env || echo "hot$n failed"
done
