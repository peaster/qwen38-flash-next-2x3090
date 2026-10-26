#!/usr/bin/env bash
# E9b: the 256K candidate at 78 slots per layer on average (hot80 failed the full-context probe on fragmentation).
set -uo pipefail
cd "$(dirname "$0")"
./session.sh runs/e9-256k-78 "full:0:1 long:0:2 decode:0:1 heldout:0:1" overrides/e9-hooks.env overrides/e9-candidate.env overrides/e9-hot78.env overrides/e9-alloc78.env || echo "256k-78 failed"
