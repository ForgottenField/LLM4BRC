#!/usr/bin/env bash
# Build the PDG builder clang tool.
#
# The output path is fixed: ``build/pdg_builder`` is the ONE binary this repo
# has, and ``tools/build_project_deps.py`` verifies the artifact's
# ``metadata.version`` against this source after every build.  An earlier
# ``build/pdg_builder_local`` (a second, older copy) survived here and was the
# binary the on-demand builder actually ran, so ``pdg_protobuf.json`` came out
# as version 1.0 — missing the macro-branch annotation and the
# postdominator-derived control dependencies — and an entire 21-report
# evaluation was run on it.  Do not introduce a second binary name.
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR="${BUILD_DIR:-build}"

# clang's AST headers live under the LLVM include prefix when libclang-14-dev
# is installed; an out-of-tree copy is used only if it actually exists (the
# hard-coded path to another user's home has silently died twice).
CLANG_DEV_HEADERS="${CLANG_DEV_HEADERS:-/usr/lib/llvm-14/include}"
if [[ ! -d "${CLANG_DEV_HEADERS}" ]]; then
  echo "ERROR: clang headers not found: ${CLANG_DEV_HEADERS}" >&2
  echo "       install libclang-14-dev, or pass CLANG_DEV_HEADERS=<dir>" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"

LLVM_CXXFLAGS="$(llvm-config-14 --cxxflags 2>/dev/null || llvm-config --cxxflags)"

echo "==> Building pdg_builder ..."

clang++-14 -o "${BUILD_DIR}/pdg_builder" \
  ${LLVM_CXXFLAGS} -std=c++17 \
  -I"${CLANG_DEV_HEADERS}" \
  -L/usr/lib/llvm-14/lib \
  PDGBuilder.cpp \
  -l:libclang-cpp.so.14 \
  -lLLVM-14

echo "==> Tool built: ${BUILD_DIR}/pdg_builder"
# Anchor to a code line: an unanchored grep reports the first mention anywhere,
# including a comment that merely quotes the assignment.
grep -E '^[[:space:]]*Metadata\["version"\]' PDGBuilder.cpp | sed 's/^ *//' \
  || echo "    WARNING: could not read Metadata[\"version\"] from PDGBuilder.cpp"
