#!/usr/bin/env bash
# One measurement session: start the server, wait for readiness, run suites, stop the server.
#
#   session.sh <run_dir> "<suite>:<warmup>:<repeat> ..." [override.env ...]
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
run_dir=${1:?}; suites=${2:?}; shift 2
mkdir -p "$run_dir"

"$here/launch.sh" "$run_dir" "$@" > "$run_dir/server.log" 2>&1 &
server=$!
trap 'docker stop -t 60 qwen38-flash-next >/dev/null 2>&1 || true; wait $server 2>/dev/null || true' EXIT

for _ in $(seq 300); do
  grep -q 'Application startup complete' "$run_dir/server.log" 2>/dev/null && break
  if ! kill -0 $server 2>/dev/null || grep -qE 'Engine core initialization failed|EngineDeadError' "$run_dir/server.log" 2>/dev/null; then
    echo "server failed during startup" >&2; grep -m3 -E 'Error' "$run_dir/server.log" >&2; exit 1
  fi
  sleep 5
done
grep -q 'Application startup complete' "$run_dir/server.log" || { echo "server not ready after 25 min" >&2; exit 1; }
python3 "$here/client.py" wait "$run_dir" --log "$run_dir/server.log"
for spec in $suites; do
  IFS=: read -r name warmup repeat <<< "$spec"
  if [[ $name == exact ]]; then python3 "$here/client.py" exact "$run_dir" --repeat "$repeat" 2>&1 | tee -a "$run_dir/client.log"; continue; fi
  python3 "$here/client.py" suite "$run_dir" "$name" --warmup "$warmup" --repeat "$repeat" 2>&1 | tee -a "$run_dir/client.log"
done
grep -E 'Traceback|OutOfMemory|preempt' "$run_dir/server.log" | head -5 || true
