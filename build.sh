#!/usr/bin/env bash
# Build rustwood (a standalone cuda-oxide project) against a sibling cuda-oxide checkout.
#
#   ./build.sh                      # build (sm_103 / B300)
#   ./build.sh --features f64-hist  # build with a feature
#   ARCH=sm_90 ./build.sh           # target a different GPU
#
# Produces target/release/rustwood.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OXIDE_REPO="$(cd "$HERE/external/cuda-oxide" && pwd)"
ARCH="${ARCH:-sm_103}"

export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export CUDA_HOME="${CUDA_HOME:-$CUDA_PATH}"
export CUDA_OXIDE_BACKEND="$OXIDE_REPO/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so"

# The codegen backend (librustc_codegen_cuda.so) is a one-time, slow build. It is its own
# workspace with a pinned toolchain (rustc-dev + llvm-tools auto-install via rustup), and it
# lands at exactly $CUDA_OXIDE_BACKEND. Build it here if absent so a fresh checkout just works.
if [[ ! -f "$CUDA_OXIDE_BACKEND" ]]; then
  echo "cuda-oxide backend not found — building it once (slow, ~10 min the first time)..." >&2
  ( cd "$OXIDE_REPO/crates/rustc-codegen-cuda" && cargo build ) || {
    echo "failed to build the cuda-oxide backend (need libclang-dev + CUDA toolkit)" >&2
    exit 1
  }
fi

cd "$HERE"
exec "$OXIDE_REPO/target/debug/cargo-oxide" build --arch "$ARCH" "$@"
