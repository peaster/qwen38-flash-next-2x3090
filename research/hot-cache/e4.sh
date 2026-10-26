#!/usr/bin/env bash
# E4 after E3: CPU compute of the hot76 LRU misses, two thread layouts.
set -uo pipefail
cd "$(dirname "$0")"
while pgrep -f "e3.sh" >/dev/null; do sleep 15; done
repo=$(cd ../.. && pwd -P)
source "$repo/.env"
model=$(cd "$MODEL_DIR" && pwd -P)
for variant in "7t:1-7:9-15" "14t:1-7,17-23:9-15,25-31"; do
  IFS=: read -r name c0 c1 <<< "$variant"
  docker run --rm --gpus all --ipc host --ulimit memlock=-1 --entrypoint python3 \
    -v "$repo:/repo" -v "$model:/model:ro" qwen38-flash-next-2x3090:locked \
    -I /repo/research/hot-cache/e4_cpu_bench.py --trace-dir /repo/research/hot-cache/runs/e0b-hooks-on \
    --capacity 76 --steps 400 --cpus0 "$c0" --cpus1 "$c1" --out "/repo/research/hot-cache/runs/e4/$name.npz" \
    > "runs/e4/$name.log" 2>&1
done
