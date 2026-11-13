#!/usr/bin/env bash
# Starts the q36 server on Linux: OpenAI-compatible API on http://127.0.0.1:9936/v1
# MODEL = the converted .q36 file (the .tok file with the same name must be next to it). Extra arguments are passed through,
# e.g.  ./start.sh --ctx 65536 --port 8080
root="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-$root/models/Qwen3.6-35B-A3B-UD-Q4_K_M.q36}"
[ -f "$MODEL" ] || { echo "Model not found: $MODEL (set MODEL or put the files into models/)"; exit 1; }
source "${ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" --force >/dev/null 2>&1 || true
exec "$root/bin/q36-server" "$MODEL" --ctx 32768 --vram-gb 10.5 --mtp 2 --port 9936 "$@"
