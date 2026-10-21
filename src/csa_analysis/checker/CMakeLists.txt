cmake_minimum_required(VERSION 3.13)
project(ReachabilityPlugin)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Find LLVM & Clang (installed system-wide) ----
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})

# Extra headers from libclang-14-dev (StaticAnalyzer headers not in
# llvm-14-dev).  Point this at your extracted .deb location.
set(EXTRACTED_CLANG_DEV
  "/home/yanghq/poc_generation/.build-deps/usr/lib/llvm-14/include")
if(EXISTS "${EXTRACTED_CLANG_DEV}/clang/StaticAnalyzer/Core/Checker.h")
  include_directories(${EXTRACTED_CLANG_DEV})
else()
  message(FATAL_ERROR
    "Missing StaticAnalyzer headers.  Install libclang-14-dev or extract the "
    ".deb to ${EXTRACTED_CLANG_DEV}")
endif()

add_definitions(${LLVM_DEFINITIONS})

# ---- Out-of-tree loadable module ----
llvm_config(ReachabilityPlugin USE_TOOLS)

add_llvm_loadable_module(ReachabilityPlugin
  ReachabilityChecker.cpp
  PLUGIN_TOOL clang
)
