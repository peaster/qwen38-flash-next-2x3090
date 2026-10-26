#!/usr/bin/env bash
# E5: in-situ CPU expert path (64 GB profile) on this host, default slicing then 5 slices.
set -uo pipefail
cd "$(dirname "$0")"
./session.sh runs/e5-64gb-s1 "decode:0:2" ../../configs/2x3090-64gb.env overrides/e5-64gb.env || echo "s1 failed"
./session.sh runs/e5-64gb-s5 "decode:0:2" ../../configs/2x3090-64gb.env overrides/e5-64gb.env overrides/e5-slices5.env || echo "s5 failed"
