//===- ReachabilityChecker.cpp -- CSA plugin for guided path extraction ---===//
//
// This checker watches CSA's symbolic execution and emits a bug report
// whenever execution reaches a user-specified source location (file:line).
//
// Configuration (via -analyzer-config):
//   reachability.ReachabilityChecker:TargetFile=<path>
//   reachability.ReachabilityChecker:TargetLine=<line>
//===----------------------------------------------------------------------===//

#include "clang/Analysis/CFG.h"
#include "clang/Analysis/ProgramPoint.h"
#include "clang/Basic/SourceManager.h"
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include <memory>

using namespace clang;
using namespace ento;

// ---------------------------------------------------------------------------
// ReachabilityChecker
// ---------------------------------------------------------------------------
namespace {

class ReachabilityChecker final
    : public Checker<check::PreStmt<Stmt>> {

  mutable std::string TargetFileStr;
  mutable unsigned TargetLine = 0;
  mutable std::unique_ptr<BugType> BT;

public:
  void checkPreStmt(const Stmt *S, CheckerContext &C) const {
    // ---- lazy initialisation from -analyzer-config ----
    if (TargetLine == 0) {
      AnalyzerOptions &Opts =
          C.getAnalysisManager().getAnalyzerOptions();
      llvm::StringRef FileRef =
          Opts.getCheckerStringOption(
              static_cast<const CheckerBase *>(this), "TargetFile");
      TargetFileStr = FileRef.str();
      llvm::StringRef LineRef =
          Opts.getCheckerStringOption(
              static_cast<const CheckerBase *>(this), "TargetLine");
      if (LineRef.getAsInteger(10, TargetLine))
        TargetLine = 0;
    }

    if (TargetLine == 0 || TargetFileStr.empty())
      return;

    const SourceManager &SM = C.getSourceManager();
    const SourceLocation Loc = S->getBeginLoc();

    if (!Loc.isValid() || SM.isInSystemHeader(Loc) ||
        SM.isInExternCSystemHeader(Loc))
      return;

    const unsigned Line = SM.getSpellingLineNumber(Loc);
    if (Line != TargetLine)
      return;

    const llvm::StringRef FileName = SM.getFilename(Loc);
    if (FileName.empty() ||
        (!FileName.endswith(TargetFileStr) &&
         FileName.find(TargetFileStr) == llvm::StringRef::npos))
      return;

    // ---- target reached ----
    if (!BT) {
      BT = std::make_unique<BugType>(
          this, "Reachability", "Reachability Analysis");
    }

    ExplodedNode *ErrNode = C.generateNonFatalErrorNode(C.getState());
    if (!ErrNode)
      return;

    auto R = std::make_unique<PathSensitiveBugReport>(
        *BT,
        "Execution reaches target location at " + TargetFileStr + ":" +
            llvm::Twine(TargetLine).str(),
        ErrNode);
    C.emitReport(std::move(R));
  }
};

} // end anonymous namespace

// ---------------------------------------------------------------------------
// Plugin entry points
// ---------------------------------------------------------------------------
extern "C" void clang_registerCheckers(CheckerRegistry &registry) {
  registry.addChecker<ReachabilityChecker>(
      "reachability.ReachabilityChecker",
      "Traces execution paths to a specified source location",
      "");

  registry.addCheckerOption(
      "string",
      "reachability.ReachabilityChecker", "TargetFile",
      "", "Source file path to watch", "");
  registry.addCheckerOption(
      "string",
      "reachability.ReachabilityChecker", "TargetLine",
      "", "Line number in the target file", "");
}

extern "C" const char clang_analyzerAPIVersionString[] =
    CLANG_ANALYZER_API_VERSION_STRING;
