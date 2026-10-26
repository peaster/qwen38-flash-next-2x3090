#!/usr/bin/env bash
# E9c: margin-preserving candidates (E9 OOM'd at hot80/96). Controls: e9-f5, e9-f5b (same day).
set -uo pipefail
cd "$(dirname "$0")"
./session.sh runs/e9c-256k "full:0:2 long:0:2 decode:0:1 heldout:0:1" overrides/e9-hooks.env overrides/e9-candidate.env overrides/e8-alloc.env || echo "256k failed"
./session.sh runs/e9c-128k "long:0:3 decode:0:1 heldout:0:1" overrides/e9-hooks.env overrides/e9-candidate.env overrides/e9c-128k.env overrides/e9-alloc92.env || echo "128k failed"
