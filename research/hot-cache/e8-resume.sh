#!/usr/bin/env bash
# E8 resumed after the per-layer allocation crash (stream_stage snapshot assumed equal slot counts; fixed).
set -uo pipefail
cd "$(dirname "$0")"
s="heldout:0:2"
./session.sh runs/e8-alloc "$s" overrides/e8-base.env overrides/e8-alloc.env || echo "alloc failed"
./session.sh runs/e8-both "$s" overrides/e8-base.env overrides/e8-lrfu.env overrides/e8-alloc.env || echo "both failed"
./session.sh runs/e8-lru2 "$s" overrides/e8-base.env || echo "lru2 failed"
