#!/usr/bin/env bash
# E7: F5 control vs the reference-drop trims; exactness probe x2 and one 131K request each.
set -uo pipefail
cd "$(dirname "$0")"
./session.sh runs/e7-ctl "exact:0:2 long:0:1" overrides/e7-hooks.env || echo "ctl failed"
./session.sh runs/e7-trim "exact:0:2 long:0:1" overrides/e7-hooks.env overrides/e7-trim.env || echo "trim failed"
