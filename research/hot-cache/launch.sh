#!/usr/bin/env bash
# Research launcher: scripts/docker_serve.sh plus the observation hooks, optional overlay file overrides and
# extra vLLM arguments. Runs in the foreground; redirect its output to <run_dir>/server.log.
#
#   research/hot-cache/launch.sh <run_dir> [override.env ...]
#
# Settings come from the repository .env, then each override file in order. Research-only variables:
#   QWEN38_RESEARCH=1 (+ QWEN38_RESEARCH_* knobs, see hooks/qwen38_research.py)
#   QWEN38_RESEARCH_VLLM_ARGS   extra arguments appended to `vllm serve`
#   OVERLAY_OVERRIDES           space-separated <path relative to the vllm package>=<host file, absolute or
#                               relative to this directory>
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(cd -- "$here/../.." && pwd -P)
run_dir=${1:?usage: launch.sh <run_dir> [override.env ...]}
shift
mkdir -p "$run_dir"
run_dir=$(cd -- "$run_dir" && pwd -P)

set -a
# shellcheck source=/dev/null
source "$repo/.env"
for override in "$@"; do
  # shellcheck source=/dev/null
  source "$override"
done
set +a

image=${IMAGE:-qwen38-flash-next-2x3090:locked}
port=${PORT:-8000}

# Record exactly what was launched.
{
  echo "date=$(date -Is)"
  echo "image=$(docker image inspect --format '{{.Id}}' "$image")"
  echo "commit=$(git -C "$repo" rev-parse HEAD)"
  echo "dirty=$(git -C "$repo" status --porcelain | wc -l)"
  echo "overrides=$*"
} > "$run_dir/launch.txt"
compgen -e | grep -E '^(QWEN38_|VLLM_|MAX_|KV_|CPU_OFFLOAD|DISABLE_|PYTORCH_|MTP_|OVERLAY_|GPU_DEVICES|JIT_)' \
  | sort | while read -r name; do echo "$name=${!name}"; done > "$run_dir/env.txt"
cp "$here/hooks/qwen38_research.py" "$run_dir/hooks-qwen38_research.py"

# The image's entrypoint is scripts/serve-container.sh; add the extra-arguments hook to a copy of it.
sed 's|^exec vllm serve "\$model" \\$|exec vllm serve "$model" ${QWEN38_RESEARCH_VLLM_ARGS:-} \\|' \
  "$repo/scripts/serve-container.sh" > "$run_dir/qwen38-serve"
grep -q 'QWEN38_RESEARCH_VLLM_ARGS' "$run_dir/qwen38-serve" || { echo "serve script patch failed" >&2; exit 2; }
chmod +x "$run_dir/qwen38-serve"

docker_env=(-e "PORT=$port" -e "PYTHONPATH=/research/hooks" -e "QWEN38_RESEARCH_OUT=/research/out")
while IFS= read -r name; do
  docker_env+=(-e "$name=${!name}")
done < <(compgen -e | grep -E '^(QWEN38_[A-Z0-9_]+|VLLM_[A-Z0-9_]+|SERVED_MODEL_NAME|MAX_MODEL_LEN|MAX_NUM_BATCHED_TOKENS|MAX_PARALLEL_LOADING_WORKERS|MAX_NUM_SEQS|KV_CACHE_MEMORY_BYTES|KV_CACHE_DTYPE|CPU_OFFLOAD_GB|DISABLE_CUSTOM_ALL_REDUCE|PYTORCH_CUDA_ALLOC_CONF|ENABLE_VISION|MTP_DEPTH)$' | grep -v '^QWEN38_RESEARCH_OUT$')

mounts=(
  -v "$here/hooks:/research/hooks:ro"
  -v "$run_dir:/research/out"
  -v "$run_dir/qwen38-serve:/usr/local/bin/qwen38-serve:ro"
  -v "$repo/jit-cache/humming:/root/.humming"
  -v "$repo/jit-cache/triton:/root/.triton"
  -v "$(cd -- "$MODEL_DIR" && pwd -P):/model:ro"
)
for item in ${OVERLAY_OVERRIDES:-}; do
  rel=${item%%=*}
  src=${item#*=}
  [[ $src == /* ]] || src=$here/$src  # relative host paths are relative to this directory
  host=$(cd -- "$(dirname -- "$src")" && pwd -P)/$(basename -- "$src")
  [[ -f "$host" ]] || { echo "missing override $host" >&2; exit 2; }
  mounts+=(-v "$host:/usr/local/lib/python3.12/dist-packages/vllm/$rel:ro")
  echo "override $rel <- $host ($(sha256sum "$host" | cut -c1-16))" >> "$run_dir/launch.txt"
done

gpu_devices=${GPU_DEVICES:-all}
case "$gpu_devices" in
  \"*\") ;;
  *,*) gpu_devices="\"$gpu_devices\"" ;;
esac

# A previous --rm container may still be being removed.
for _ in $(seq 60); do
  docker container inspect qwen38-flash-next &>/dev/null || break
  sleep 1
done
exec docker run --rm \
  --name qwen38-flash-next \
  --gpus "$gpu_devices" \
  --ipc host \
  --cap-add SYS_PTRACE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "0.0.0.0:$port:$port" \
  "${docker_env[@]}" \
  "${mounts[@]}" \
  "$image" /model
