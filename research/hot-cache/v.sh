#!/usr/bin/env bash
# Release validation of the promoted overlay (new :locked image, no overlay overrides, research hooks off).
# Order: 256K profile, F5 control (same image: trims on, policies off), 128K profile.
set -uo pipefail
cd "$(dirname "$0")"
s256="full:0:2 long:0:2 decode:0:1 heldout:0:1"
./session.sh runs/v-256k "$s256" overrides/v-nohooks.env overrides/v-256k.env || echo "v-256k failed"
./session.sh runs/v-f5 "$s256" overrides/v-nohooks.env || echo "v-f5 failed"
./session.sh runs/v-128k "long:0:2 decode:0:1 heldout:0:1" overrides/v-nohooks.env overrides/v-128k.env || echo "v-128k failed"
