#!/usr/bin/env bash
# E7b: largest hot set at 256K with the trims: hot80, then hot78 if hot80 fails. Full-context probe first.
set -uo pipefail
cd "$(dirname "$0")"
s="full:0:1 long:0:1 decode:0:1"
./session.sh runs/e7b-hot80 "$s" overrides/e7-hooks.env overrides/e7-trim.env overrides/e7b-hot80.env && exit 0
echo "hot80 failed"
./session.sh runs/e7b-hot78 "$s" overrides/e7-hooks.env overrides/e7-trim.env overrides/e7b-hot78.env || echo "hot78 failed"
