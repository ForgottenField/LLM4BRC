#!/usr/bin/env bash
# Build the out-of-tree CSA Reachability plugin.
# Links dynamically against libclang-cpp to avoid duplicate LLVM symbol
# registration (CommandLine option conflicts).
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR="${BUILD_DIR:-build}"
PLUGIN_SO="${BUILD_DIR}/libReachabilityPlugin.so"
CLANG_DEV_HEADERS="/home/yanghq/poc_generation/.build-deps/usr/lib/llvm-14/include"

mkdir -p "${BUILD_DIR}"

LLVM_CXXFLAGS="$(llvm-config-14 --cxxflags)"

echo "==> Building reachability checker plugin (dynamic: -lclang-cpp) ..."

clang++-14 -shared -fPIC -o "${PLUGIN_SO}" \
  ${LLVM_CXXFLAGS} \
  -I/usr/lib/llvm-14/include \
  -I"${CLANG_DEV_HEADERS}" \
  ReachabilityChecker.cpp \
  -lclang-cpp

echo "==> Plugin built: ${PLUGIN_SO}"
ls -lh "${PLUGIN_SO}"
