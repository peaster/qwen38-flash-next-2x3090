#!/usr/bin/env bash
# E8: cache policies in the runtime on held-out prompts (F5, 256K, hot76 total). heldout x2 per session.
# Order: released LRU, LRFU, per-layer allocation, both, released LRU again (drift check).
set -uo pipefail
cd "$(dirname "$0")"
s="heldout:0:2"
./session.sh runs/e8-lru "$s" overrides/e8-base.env || echo "lru failed"
./session.sh runs/e8-lrfu "$s" overrides/e8-base.env overrides/e8-lrfu.env || echo "lrfu failed"
./session.sh runs/e8-alloc "$s" overrides/e8-base.env overrides/e8-alloc.env || echo "alloc failed"
./session.sh runs/e8-both "$s" overrides/e8-base.env overrides/e8-lrfu.env overrides/e8-alloc.env || echo "both failed"
./session.sh runs/e8-lru2 "$s" overrides/e8-base.env || echo "lru2 failed"
