#!/usr/bin/env bash
# Builds q36 on Linux. Output: build/ and bin/ (q36-server, q36, pack, mktok).
# Requirements: Intel oneAPI DPC++/C++ Compiler (icpx) 2025.x or newer, g++ with C++17,
# Intel GPU driver stack with Level Zero (compute-runtime) that supports the Arc B580.
# ONEAPI_ROOT: oneAPI install folder (default /opt/intel/oneapi). CXX: compiler for the CPU-only tools (default g++).
set -eo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
[ -f "$ONEAPI_ROOT/setvars.sh" ] || { echo "oneAPI not found in $ONEAPI_ROOT (set ONEAPI_ROOT)"; exit 1; }
source "$ONEAPI_ROOT/setvars.sh" --force >/dev/null 2>&1 || true
command -v icpx >/dev/null || { echo "icpx not found after sourcing $ONEAPI_ROOT/setvars.sh"; exit 1; }
CXX="${CXX:-g++}"
mkdir -p "$root/build" "$root/bin"; cd "$root/build"

# GPU programs: SYCL + ESIMD, kernels compiled ahead of time for the Arc B580 (Xe2, bmg-g21)
GPUFLAGS=(-fsycl -O2 -std=c++17 -mavx2 -mfma -mf16c -fsycl-targets=spir64_gen -Xs "-device bmg-g21")
echo "[1/4] q36-server"; icpx "${GPUFLAGS[@]}" ../tools/server.cpp -o q36-server -lpthread
echo "[2/4] q36 (CLI)";  icpx "${GPUFLAGS[@]}" ../tools/q36.cpp -o q36 -lpthread
# CPU-only tools
echo "[3/4] pack";  $CXX -O2 -std=c++17 -mavx2 -mfma -mf16c -pthread ../tools/pack.cpp -o pack
echo "[4/4] mktok"; $CXX -O2 -std=c++17 -mavx2 -mfma -mf16c -pthread ../tools/mktok.cpp -o mktok

cp q36-server q36 pack mktok "$root/bin/"
echo "build ok: $root/bin"
