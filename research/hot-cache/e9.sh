#!/usr/bin/env bash
# E9: combined candidates vs F5, same day. Order: F5, 256K candidate, 128K candidate, F5 again (drift).
set -uo pipefail
cd "$(dirname "$0")"
s256="full:0:1 long:0:2 decode:0:1 heldout:0:1"
s128="long:0:2 decode:0:1 heldout:0:1"
./session.sh runs/e9-f5 "$s256" overrides/e9-hooks.env || echo "f5 failed"
./session.sh runs/e9-256k "$s256" overrides/e9-hooks.env overrides/e9-candidate.env overrides/e7b-hot80.env overrides/e9-alloc80.env || echo "256k failed"
./session.sh runs/e9-128k "$s128" overrides/e9-hooks.env overrides/e9-candidate.env overrides/e9-128k.env overrides/e9-alloc96.env || echo "128k failed"
./session.sh runs/e9-f5b "$s256" overrides/e9-hooks.env || echo "f5b failed"
