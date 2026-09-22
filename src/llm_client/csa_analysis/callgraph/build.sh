#!/usr/bin/env bash
# Build the call graph builder clang tool.
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR="${BUILD_DIR:-build}"
CLANG_DEV_HEADERS="/home/yanghq/poc_generation/.build-deps/usr/lib/llvm-14/include"

mkdir -p "${BUILD_DIR}"

LLVM_CXXFLAGS="$(llvm-config-14 --cxxflags 2>/dev/null || llvm-config --cxxflags)"

echo "==> Building call_graph_builder ..."

clang++-14 -o "${BUILD_DIR}/call_graph_builder" \
  ${LLVM_CXXFLAGS} \
  -I/usr/lib/llvm-14/include \
  -I"${CLANG_DEV_HEADERS}" \
  -L/usr/lib/llvm-14/lib \
  CallGraphBuilder.cpp \
  -l:libclang-cpp.so.14 \
  -lLLVM-14

echo "==> Tool built: ${BUILD_DIR}/call_graph_builder"
