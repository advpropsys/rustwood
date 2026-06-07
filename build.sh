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
OXIDE_REPO="$(cd "$HERE/../cuda-oxide" && pwd)"
ARCH="${ARCH:-sm_103}"

export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export CUDA_HOME="${CUDA_HOME:-$CUDA_PATH}"
export CUDA_OXIDE_BACKEND="$OXIDE_REPO/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so"

if [[ ! -f "$CUDA_OXIDE_BACKEND" ]]; then
  echo "backend not found: $CUDA_OXIDE_BACKEND" >&2
  echo "build it once with: (cd $OXIDE_REPO && cargo oxide doctor)" >&2
  exit 1
fi

cd "$HERE"
exec "$OXIDE_REPO/target/debug/cargo-oxide" build --arch "$ARCH" "$@"
