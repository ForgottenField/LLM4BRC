//===- PDGBuilder.cpp -- Clang tool to build Program Dependence Graphs ------===//
//
// Builds a per-function Program Dependence Graph (PDG) capturing:
//   1. PDG nodes    — statement-level AST node representations
//   2. Def-use edges — data dependence (via reaching definitions on the CFG)
//   3. Control edges — control dependence (via post-dominator tree)
//   4. Summary      — input/output variable sets for cross-function propagation
//
// Outputs a JSON file consumed by the Python backward-slicing engine.
//
// Architecture follows callgraph/CallGraphBuilder.cpp:
//   libTooling -> JSONCompilationDatabase -> ClangTool ->
//   FrontendActionFactory -> ASTFrontendAction -> ASTConsumer ->
//   RecursiveASTVisitor
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/CFG.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/JSONCompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace clang;

// ===========================================================================
// Data structures matching the Python-side pdg_models.py contract
// ===========================================================================

/// One statement-level node in a function's PDG.
struct PDGNodeInfo {
  std::string id;         // "S_<line>_<col>[_<disambig>]"
  std::string kind;       // "decl", "assign", "call_expr", "if_cond", ...
  std::string variable;   // variable defined (empty if none)
  std::string expression; // textual expression
  std::string type;       // C++ type string
  unsigned sourceLine = 0;
  unsigned sourceCol = 0;
  unsigned blockId = 0;
  bool isCall = false;
  std::string callee;
  std::vector<std::string> actualParams;

  // ---- Macro-expansion provenance ----
  // sourceLine/sourceCol above are *expansion* (call-site) coordinates, as
  // required for downstream line-range matching.  The fields below record
  // where the text actually came from, so macro-generated statements and
  // branches can be told apart from hand-written ones.
  bool isMacro = false;            // text came from a macro expansion
  std::string macroName;           // immediate macro name ("" if not a macro)
  std::string macroDefFile;        // file the macro is defined in (spelling)
  unsigned macroDefLine = 0;       // line the macro is defined / expanded at
};

/// Def-use edge: data flows from def to use.
struct DefUseEdgeInfo {
  std::string defNodeId;
  std::string useNodeId;
  std::string variable;
};

/// Control-dependence edge: predicate -> dependent statement.
struct ControlDepEdgeInfo {
  std::string sourceId;
  std::string targetId;
  std::string kind; // "if_branch", "loop_body", "loop_exit", "switch_case"
  // True when the predicate (*source*) is text produced by a macro expansion,
  // i.e. the branch exists only because a macro was expanded here.  This is
  // the per-edge form of the "macro branch" flag; it lets a consumer count
  // macro branches without re-resolving the predicate node.
  bool isMacroBranch = false;
};

/// One control-dependence predicate that originates from a macro expansion.
/// Emitted per function so the expanded branches can be enumerated (and
/// counted per macro) without walking nodes and edges.
struct MacroBranchInfo {
  std::string nodeId;        // predicate node ID (S_<line>_<col>...)
  std::string macroName;     // immediate macro name, e.g. "READANDCHECK"
  std::string macroDefFile;  // file the macro body is spelled in
  unsigned macroDefLine = 0; // spelling line inside that file
  unsigned callSiteLine = 0; // expansion (call-site) line in this function
  std::string expression;    // expanded predicate text, e.g. "!(ret == (1))"
  unsigned dependentCount = 0; // statements control-dependent on it
};

/// Per-function PDG.
struct FunctionPDGInfo {
  std::string functionName;
  std::string sourceFile;
  std::vector<PDGNodeInfo> nodes;
  std::vector<DefUseEdgeInfo> defUseEdges;
  std::vector<ControlDepEdgeInfo> controlDepEdges;
  std::vector<std::string> inputVariables;
  std::vector<std::string> outputVariables;

  // ---- Macro-expansion statistics ----
  std::vector<MacroBranchInfo> macroBranches;       // enumerated macro branches
  std::map<std::string, unsigned> macroBranchByName; // macro name -> #branches
};

// ===========================================================================
// Global helpers (set per translation unit)
// ===========================================================================

static SourceManager *GSM = nullptr;
static const LangOptions *GLO = nullptr;

static std::string getSF(SourceLocation Loc) {
  if (Loc.isInvalid() || !GSM) return "";
  return GSM->getFilename(Loc).str();
}

static unsigned getSL(SourceLocation Loc) {
  if (Loc.isInvalid() || !GSM) return 0;
  // Use expansion line number so that macro-invocation statements are
  // attributed to the call site, not to the macro definition body.
  return GSM->getExpansionLineNumber(Loc);
}

static unsigned getSC(SourceLocation Loc) {
  if (Loc.isInvalid() || !GSM) return 0;
  // Ditto — expansion column for macro-invocation sites.
  return GSM->getExpansionColumnNumber(Loc);
}

/// True when ``Loc`` belongs to a macro expansion.
///
/// ``isMacroID()`` is true both for tokens written in the macro body and for
/// tokens that arrived as macro arguments, which is exactly the set we want:
/// in both cases the statement would not exist in the source without the
/// macro invocation.
static bool isMacroLoc(SourceLocation Loc) {
  return Loc.isValid() && Loc.isMacroID();
}

/// Immediate macro name for a macro-expansion location ("" when not a macro).
static std::string getMacroName(SourceLocation Loc) {
  if (!GSM || !GLO || !isMacroLoc(Loc)) return "";
  return Lexer::getImmediateMacroName(Loc, *GSM, *GLO).str();
}

/// Spelling (macro-definition) file/line for a macro-expansion location.
/// Leaves the outputs untouched when ``Loc`` is not a macro expansion.
static void getMacroDefLoc(SourceLocation Loc, std::string &File,
                           unsigned &Line) {
  if (!GSM || !isMacroLoc(Loc)) return;
  SourceLocation Spell = GSM->getSpellingLoc(Loc);
  if (Spell.isInvalid()) return;
  File = GSM->getFilename(Spell).str();
  Line = GSM->getSpellingLineNumber(Spell);
}

/// Fill the macro-provenance fields of a node from a statement location.
static void annotateMacro(PDGNodeInfo &N, SourceLocation Loc) {
  N.isMacro = isMacroLoc(Loc);
  if (!N.isMacro) return;
  N.macroName = getMacroName(Loc);
  getMacroDefLoc(Loc, N.macroDefFile, N.macroDefLine);
}

/// The DeclStmt of the condition variable declared by an ``if``/``while``/
/// ``switch`` terminator, if it has one.
///
/// ``if (const X *p = dynamic_cast<const X *>(o))`` — what the TRYCLONE macro
/// expands to — is a *condition variable declaration*.  Clang's terminator
/// condition for it is a bare read of ``p``, while the informative form of the
/// branch test is the declaration that performs the dynamic_cast.  Returns
/// nullptr for terminators that declare no condition variable.
static const DeclStmt *conditionVariableDeclStmt(const Stmt *Term) {
  if (auto *IS = dyn_cast_or_null<IfStmt>(Term))
    return IS->getConditionVariableDeclStmt();
  if (auto *WS = dyn_cast_or_null<WhileStmt>(Term))
    return WS->getConditionVariableDeclStmt();
  if (auto *SS = dyn_cast_or_null<SwitchStmt>(Term))
    return SS->getConditionVariableDeclStmt();
  return nullptr;
}

/// Get the variable name from a ValueDecl.
static std::string varName(const ValueDecl *VD) {
  if (!VD) return "";
  return VD->getNameAsString();
}

/// Extract the variable (name, type) defined by a statement, if any.
static std::pair<std::string, std::string> extractDef(const Stmt *S) {
  if (!S) return {};

  if (auto *DS = dyn_cast<DeclStmt>(S)) {
    for (auto *D : DS->decls()) {
      if (auto *VD = dyn_cast<VarDecl>(D))
        return {VD->getNameAsString(), VD->getType().getAsString()};
    }
    return {};
  }

  if (auto *BO = dyn_cast<BinaryOperator>(S)) {
    if (BO->isAssignmentOp()) {
      auto *LHS = BO->getLHS()->IgnoreParenImpCasts();
      if (auto *DRE = dyn_cast<DeclRefExpr>(LHS))
        return {varName(DRE->getDecl()), DRE->getType().getAsString()};
    }
    return {};
  }

  if (auto *UO = dyn_cast<UnaryOperator>(S)) {
    if (UO->isIncrementOp() || UO->isDecrementOp()) {
      auto *Sub = UO->getSubExpr()->IgnoreParenImpCasts();
      if (auto *DRE = dyn_cast<DeclRefExpr>(Sub))
        return {varName(DRE->getDecl()), DRE->getType().getAsString()};
    }
    return {};
  }

  return {};
}

/// Recursively collect variable names *used* (read) by a statement.
static void collectUses(const Stmt *S, std::vector<std::string> &out) {
  if (!S) return;

  // For expressions, strip implicit casts/parens to find the underlying
  // variable references.
  if (auto *E = dyn_cast<Expr>(S)) {
    E = E->IgnoreParenImpCasts();

    // Direct variable reference
    if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
      std::string n = varName(DRE->getDecl());
      if (!n.empty()) out.push_back(n);
      return;
    }

    // Function call: collect from arguments and callee
    if (auto *CE = dyn_cast<CallExpr>(E)) {
      for (auto *Arg : CE->arguments())
        collectUses(Arg, out);
      collectUses(CE->getCallee(), out);
      return;
    }

    // Member expression: base object
    if (auto *ME = dyn_cast<MemberExpr>(E)) {
      collectUses(ME->getBase(), out);
      return;
    }

    // Array subscript: base and index
    if (auto *ASE = dyn_cast<ArraySubscriptExpr>(E)) {
      collectUses(ASE->getBase(), out);
      collectUses(ASE->getIdx(), out);
      return;
    }

    // Binary operators
    if (auto *BO = dyn_cast<BinaryOperator>(E)) {
      collectUses(BO->getLHS(), out);
      collectUses(BO->getRHS(), out);
      return;
    }

    // Unary operators
    if (auto *UO = dyn_cast<UnaryOperator>(E)) {
      collectUses(UO->getSubExpr(), out);
      return;
    }

    // Ternary conditional
    if (auto *CO = dyn_cast<ConditionalOperator>(E)) {
      collectUses(CO->getCond(), out);
      collectUses(CO->getTrueExpr(), out);
      collectUses(CO->getFalseExpr(), out);
      return;
    }

    // Other expressions (literals, this, etc.): no variables
    return;
  }

  // Non-expression statements — extract from their sub-expressions
  if (auto *IS = dyn_cast<IfStmt>(S))    { collectUses(IS->getCond(), out); return; }
  if (auto *WS = dyn_cast<WhileStmt>(S)) { collectUses(WS->getCond(), out); return; }
  if (auto *FS = dyn_cast<ForStmt>(S))   {
    if (FS->getCond()) collectUses(FS->getCond(), out);
    return;
  }
  if (auto *RS = dyn_cast<ReturnStmt>(S)) {
    if (RS->getRetValue()) collectUses(RS->getRetValue(), out);
    return;
  }
}

/// Convenience wrapper.
static std::vector<std::string> collectUsed(const Stmt *S) {
  std::vector<std::string> out;
  collectUses(S, out);
  return out;
}

/// Get callee name from a CallExpr.
static std::string getCallee(const CallExpr *CE) {
  if (!CE) return "";
  if (auto *ND = dyn_cast_or_null<NamedDecl>(CE->getCalleeDecl()))
    return ND->getNameAsString();
  return "";
}

/// Collect actual-parameter variable names from a CallExpr.
static std::vector<std::string> collectParams(const CallExpr *CE) {
  std::vector<std::string> params;
  if (!CE) return params;
  for (auto *Arg : CE->arguments()) {
    auto v = collectUsed(Arg);
    if (!v.empty()) params.push_back(v.front());
    else params.push_back("?");
  }
  return params;
}

/// Generate a unique PDG node ID.
static std::string mkNodeID(unsigned line, unsigned col,
                            llvm::StringSet<> &used) {
  std::string id = ("S_" + llvm::Twine(line) + "_" + llvm::Twine(col)).str();
  if (!used.contains(id)) { used.insert(id); return id; }
  for (unsigned i = 1; i < 999; ++i) {
    std::string c = ("S_" + llvm::Twine(line) + "_" + llvm::Twine(col) +
                     "_" + llvm::Twine(i)).str();
    if (!used.contains(c)) { used.insert(c); return c; }
  }
  return id;
}

/// Classify a statement's kind for the PDG node.
static std::string stmtKind(const Stmt *S) {
  if (!S) return "event";
  if (isa<DeclStmt>(S)) {
    for (auto *D : cast<DeclStmt>(S)->decls())
      if (isa<FunctionDecl>(D)) return "func_decl";
    return "decl";
  }
  if (auto *BO = dyn_cast<BinaryOperator>(S))
    return BO->isAssignmentOp() ? "assign" : "binary_op";
  if (isa<CallExpr>(S))       return "call_expr";
  if (isa<IfStmt>(S))         return "if_cond";
  if (isa<WhileStmt>(S))      return "while_cond";
  if (isa<ForStmt>(S))        return "for_cond";
  if (isa<ReturnStmt>(S))     return "return";
  if (isa<UnaryOperator>(S))  return "unary_op";
  if (isa<DeclRefExpr>(S))    return "declref";
  if (isa<MemberExpr>(S))     return "member_expr";
  if (isa<ArraySubscriptExpr>(S)) return "array_subscript";
  return "event";
}

// ===========================================================================
// Per-function PDG construction
// ===========================================================================

static void buildFunctionPDG(
    FunctionDecl *FD, FunctionPDGInfo &Info) {

  Stmt *Body = FD->getBody();
  if (!Body) return;

  // ---- Build CFG ----
  CFG::BuildOptions Opts;
  Opts.AddInitializers = true;
  Opts.AddCXXDefaultInitExprInAggregates = true;
  auto CFG = CFG::buildCFG(FD, Body, &FD->getASTContext(), Opts);
  if (!CFG) return;

  llvm::StringSet<> usedNodeIds;

  // ---- Phase 1: Create PDG nodes ----
  // We create one PDG node per CFGStmt element in the CFG.
  // Strategy 2 (Resilience): RecoveryExpr nodes are still created as
  // nodes with kind "recovery", and processing continues to subsequent
  // statements rather than aborting the function.
  std::map<const Stmt *, std::string> stmtToNid;
  unsigned recoveryCount = 0;

  for (const auto *B : *CFG) {
    if (!B) continue;
    unsigned bid = B->getBlockID();

    for (const auto &Elem : *B) {
      auto SE = Elem.getAs<CFGStmt>();
      if (!SE) continue;
      const Stmt *S = SE->getStmt();
      if (!S) continue;

      SourceLocation Loc = S->getBeginLoc();
      unsigned line = getSL(Loc);
      unsigned col = getSC(Loc);

      // --- Strategy 2: RecoveryExpr resilience ---
      // When we encounter a RecoveryExpr (from template instantiation
      // failure or missing headers), create a node with kind "recovery"
      // instead of failing. Still extract whatever info we can.
      PDGNodeInfo N;
      bool isRecovery = isa<RecoveryExpr>(S);

      if (isRecovery) {
        ++recoveryCount;
        // For RecoveryExpr, source location may be invalid; try end loc.
        if (line == 0) {
          Loc = S->getEndLoc();
          line = getSL(Loc);
          col = getSC(Loc);
        }
        if (line == 0) { line = 1; col = 1; } // Ultima ratio
        std::string nid = mkNodeID(line, col, usedNodeIds);
        stmtToNid[S] = nid;
        N.id = nid;
        N.kind = "recovery";
        N.sourceLine = line;
        N.sourceCol = col;
        N.blockId = bid;
        annotateMacro(N, S->getBeginLoc());
        llvm::raw_string_ostream OS(N.expression);
        S->printPretty(OS, nullptr, PrintingPolicy(*GLO));
        OS.flush();
        // No truncation here — see the "1.3" note at the end of main().
        // Mark as recovery-expr
        if (N.expression.find("recovery-expr") == std::string::npos)
          N.expression = "<recovery-expr>(" + N.expression + ")";
        // Try to extract from sub-expressions (via children iterator)
        // Note: RecoveryExpr::children() is non-const in LLVM 14
        for (auto *Child : const_cast<Stmt*>(S)->children()) {
          if (!Child) continue;
          auto [dv, dt] = extractDef(Child);
          if (!dv.empty()) { N.variable = dv; N.type = dt; break; }
        }
      } else {
        // --- Normal node creation ---
        std::string nid = mkNodeID(line, col, usedNodeIds);
        stmtToNid[S] = nid;
        N.id = nid;
        N.kind = stmtKind(S);
        N.sourceLine = line;
        N.sourceCol = col;
        N.blockId = bid;
        annotateMacro(N, Loc);

        auto [dv, dt] = extractDef(S);
        N.variable = dv;
        N.type = dt;

        // Expression text — full, untruncated (see "1.3" in main()).
        llvm::raw_string_ostream OS(N.expression);
        S->printPretty(OS, nullptr, PrintingPolicy(*GLO));
        OS.flush();

        // Call info
        if (auto *CE = dyn_cast<CallExpr>(S)) {
          N.isCall = true;
          N.callee = getCallee(CE);
          N.actualParams = collectParams(CE);
        }
      }

      Info.nodes.push_back(std::move(N));
    }
  }

  // ---- Phase 1b: AST fallback for CFG-skipped CallExpr ----
  // Clang's CFG skips the statement immediately after encountering a
  // RecoveryExpr (a known Clang CFG limitation; LLVM PR 49772).  Walk
  // the function body AST to find CallExpr that the CFG missed, and
  // create PDG nodes for them. Their def-use edges are synthesized
  // separately in Phase 2c below.
  llvm::DenseSet<unsigned> coveredLines;
  for (const auto &N : Info.nodes)
    if (N.sourceLine > 0)
      coveredLines.insert(N.sourceLine);

  std::vector<const Stmt *> recoveredStmts;
  {
    struct MissingCallFinder : RecursiveASTVisitor<MissingCallFinder> {
      const std::map<const Stmt *, std::string> &StmtToNid;
      llvm::DenseSet<unsigned> &CoveredLines;
      std::vector<const Stmt *> &Recovered;
      MissingCallFinder(const std::map<const Stmt *, std::string> &S2N,
                        llvm::DenseSet<unsigned> &CL,
                        std::vector<const Stmt *> &Rec)
          : StmtToNid(S2N), CoveredLines(CL), Recovered(Rec) {}

      bool VisitCallExpr(CallExpr *CE) {
        // Already captured by the CFG — skip
        if (StmtToNid.count(CE)) return true;
        SourceLocation Loc = CE->getBeginLoc();
        unsigned line = GSM->getExpansionLineNumber(Loc);
        // Sub-expression of a CFG-covered statement (same source line)
        if (line > 0 && CoveredLines.count(line)) return true;
        Recovered.push_back(CE);
        CoveredLines.insert(line);
        return true;
      }
    };

    MissingCallFinder Finder(stmtToNid, coveredLines, recoveredStmts);
    Finder.TraverseStmt(Body);
  }

  // Create PDG nodes for CFG-missing statements
  for (const Stmt *S : recoveredStmts) {
    auto *CE = dyn_cast<CallExpr>(S);
    if (!CE) continue;

    SourceLocation Loc = CE->getBeginLoc();
    unsigned line = getSL(Loc);
    unsigned col = getSC(Loc);
    if (line == 0) { line = 1; col = 1; }

    std::string nid = mkNodeID(line, col, usedNodeIds);
    stmtToNid[S] = nid;

    PDGNodeInfo N;
    N.id = nid;
    N.kind = "call_expr";
    N.sourceLine = line;
    N.sourceCol = col;
    // Not in CFG; use block 0 so the node is treated as
    // top-level (unconditional within the function body).
    N.blockId = 0;
    annotateMacro(N, Loc);
    {
      // Full expression — see "1.3" in main().
      llvm::raw_string_ostream OS(N.expression);
      S->printPretty(OS, nullptr, PrintingPolicy(*GLO));
      OS.flush();
    }
    N.isCall = true;
    N.callee = getCallee(CE);
    N.actualParams = collectParams(CE);

    Info.nodes.push_back(std::move(N));
  }

  if (Info.nodes.empty()) return;

  // ---- Pre-Phase 2: Synthetic parameter entry nodes ----
  // Create synthetic PDG nodes for each formal parameter with
  // source_line=0 and kind="param".  These serve as the function-entry
  // "definition" of each parameter, enabling BTBS Phase 4 reverse
  // propagation (callee→caller) when the backward slicer reaches a
  // formal parameter.
  //
  // The reaching definitions map is pre-seeded with these nodes so that
  // the first use of each parameter in the function body naturally gets
  // a def-use edge pointing to the synthetic parameter node.
  //
  // Node IDs use the prefix "P_" (e.g. "P_first", "P_plopt") to
  // distinguish them from regular "S_<line>_<col>" nodes.
  std::vector<std::string> paramNames;
  std::vector<std::string> paramNodeIds;
  for (auto *PD : FD->parameters()) {
    if (!PD) continue;
    std::string pn = PD->getNameAsString();
    if (pn.empty()) continue;
    // Skip compiler-generated parameters (e.g. VTT, __vtt_parm)
    if (pn.size() > 2 && pn[0] == '_' && pn[1] == '_') continue;

    std::string paramNid = "P_" + pn;

    PDGNodeInfo P;
    P.id = paramNid;
    P.kind = "param";
    P.variable = pn;
    P.expression = pn;
    P.type = PD->getType().getAsString();
    P.sourceLine = 0;
    P.sourceCol = 0;
    P.blockId = 0;
    P.isCall = false;
    Info.nodes.push_back(std::move(P));

    paramNames.push_back(pn);
    paramNodeIds.push_back(paramNid);
  }

  // ---- Phase 2: Reaching definitions → def-use edges ----
  // Per-block statement lists for data-flow analysis.
  std::vector<std::vector<const Stmt *>> blockStmts(CFG->getNumBlockIDs());
  std::vector<std::vector<std::pair<std::string, std::string>>> blockDefs(CFG->getNumBlockIDs());

  for (const auto *B : *CFG) {
    if (!B) continue;
    unsigned bid = B->getBlockID();
    for (const auto &Elem : *B) {
      auto SE = Elem.getAs<CFGStmt>();
      if (!SE) continue;
      const Stmt *S = SE->getStmt();
      if (!S) continue;
      blockStmts[bid].push_back(S);
      auto [dv, dt] = extractDef(S);
      if (!dv.empty())
        blockDefs[bid].emplace_back(dv, dt);
    }
  }

  // Simple reaching definitions:
  // For each block, track (var→nodeId) of reaching definitions.
  // Forward data-flow.
  std::map<std::string, std::string> reaching; // var → nodeId (global)
  // Pre-seed with synthetic parameter defs so that the first use of
  // each parameter in the function body has a reaching definition.
  for (size_t pi = 0; pi < paramNames.size(); ++pi)
    reaching[paramNames[pi]] = paramNodeIds[pi];
  std::set<std::tuple<std::string, std::string, std::string>> seenDue;

  // Process blocks in a topological-like order by following successors,
  // but also iterate for stability.
  bool changed = true;
  for (int iter = 0; changed && iter < 100; ++iter) {
    changed = false;

    for (const auto *B : *CFG) {
      if (!B) continue;
      unsigned bid = B->getBlockID();

      // Merge reaching defs from all predecessors
      std::map<std::string, std::string> entry;
      for (auto PI = B->pred_begin(), PE = B->pred_end(); PI != PE; ++PI) {
        const CFGBlock *Pred = *PI;
        if (!Pred) continue;
        // We use the global 'reaching' map — after processing a predecessor,
        // its output is stored there. For simplicity, don't maintain per-block
        // IN/OUT but just merge all known defs at block entry.
        // (Conservative: may over-approximate but safe for slicing.)
        for (auto &[v, nid] : reaching)
          entry[v] = nid;
      }

      // Scan statements within the block
      for (const Stmt *S : blockStmts[bid]) {
        auto it = stmtToNid.find(S);
        if (it == stmtToNid.end()) continue;
        const std::string &useNid = it->second;

        // For each used variable, emit def-use edge to reaching def
        for (const auto &v : collectUsed(S)) {
          if (v.empty()) continue;
          auto rd = entry.find(v);
          if (rd != entry.end() && !rd->second.empty()) {
            auto key = std::make_tuple(rd->second, useNid, v);
            if (seenDue.insert(key).second) {
              Info.defUseEdges.push_back({rd->second, useNid, v});
            }
          } else {
            // Also check global reaching (for cross-block reaching)
            auto rg = reaching.find(v);
            if (rg != reaching.end() && !rg->second.empty()) {
              auto key = std::make_tuple(rg->second, useNid, v);
              if (seenDue.insert(key).second) {
                Info.defUseEdges.push_back({rg->second, useNid, v});
              }
            }
          }
        }

        // Update entry (and reaching) with defs from this statement
        auto [dv, dt] = extractDef(S);
        if (!dv.empty()) {
          entry[dv] = useNid;
          reaching[dv] = useNid;
        }
      }

      // Check if reaching changed for this block
      // (simplified: just track via the global map)
    }
  }

  // ---- Phase 2b: Def-use fallback (Strategy 3) ----
  // If CFG-based reaching definitions produced zero or very few edges,
  // run fallback strategies to ensure basic data-flow is captured.
  if (Info.defUseEdges.empty()) {
    // Strategy 3a: Linear-scan def-use.
    // Walk nodes in program order; track the most recent definition of
    // each variable; emit edges from that def to any later use.
    std::map<std::string, std::string> lastDef; // var -> nodeId
    for (const auto &N : Info.nodes) {
      // If this node defines a variable, record it
      if (!N.variable.empty())
        lastDef[N.variable] = N.id;
      // For call_expr nodes, also record actual params as uses
      if (N.isCall) {
        for (const auto &param : N.actualParams) {
          if (param.empty() || param == "?") continue;
          auto ld = lastDef.find(param);
          if (ld != lastDef.end() && !ld->second.empty()) {
            auto key = std::make_tuple(ld->second, N.id, param);
            if (seenDue.insert(key).second)
              Info.defUseEdges.push_back({ld->second, N.id, param});
          }
        }
      }
    }
  }

  // If STILL no def-use edges, try Strategy 3b: AST-level direct def/use.
  // This is the most conservative fallback: scan all nodes and create edges
  // when a variable is referenced in an expression and also defined elsewhere.
  if (Info.defUseEdges.empty() && Info.nodes.size() > 2) {
    std::map<std::string, std::string> anyDef;
    for (const auto &N : Info.nodes)
      if (!N.variable.empty())
        anyDef[N.variable] = N.id;

    for (const auto &N : Info.nodes) {
      if (N.isCall) {
        for (const auto &param : N.actualParams) {
          if (param.empty() || param == "?") continue;
          auto ad = anyDef.find(param);
          if (ad != anyDef.end() && ad->second != N.id) {
            auto key = std::make_tuple(ad->second, N.id, param);
            if (seenDue.insert(key).second)
              Info.defUseEdges.push_back({ad->second, N.id, param});
          }
        }
      }
    }
  }

  // ---- Phase 2c: Def-use edges for AST-recovered nodes ----
  // Statements recovered in Phase 1b weren't processed by the
  // CFG-based reaching definitions algorithm above.  Create def-use
  // edges using the reaching definitions available at the final
  // function-exit state (i.e., after all CFG blocks have been
  // processed during Phase 2).
  for (const Stmt *S : recoveredStmts) {
    auto it = stmtToNid.find(S);
    if (it == stmtToNid.end()) continue;
    const std::string &useNid = it->second;

    for (const auto &v : collectUsed(S)) {
      if (v.empty()) continue;
      auto rd = reaching.find(v);
      if (rd != reaching.end() && !rd->second.empty()) {
        auto key = std::make_tuple(rd->second, useNid, v);
        if (seenDue.insert(key).second) {
          Info.defUseEdges.push_back({rd->second, useNid, v});
        }
      }
    }
  }

  // ---- Phase 3: Control dependence via post-dominator tree ----
  // Node Y is control-dependent on P if:
  //   * P has >= 2 successors
  //   * Y post-dominates one successor of P but not P itself
  //
  // Compute post-dominators via iterative data-flow on reverse CFG.

  // Collect valid block IDs
  std::vector<unsigned> blockIds;
  for (const auto *B : *CFG) {
    if (B) blockIds.push_back(B->getBlockID());
  }
  if (blockIds.empty()) return;

  // The exit block is the unique root of the post-dominator lattice.
  //
  // This used to be "the first block without successors", which is wrong as
  // soon as the function has more than one such block: clang does connect
  // `return` to the exit block, but a block ending in an infinite loop or
  // `__builtin_unreachable` has no successors either, and picking one of those
  // as the lattice root corrupts every post-dominator set in the function.
  // CFG::getExit() is the block clang itself designates as the exit.
  unsigned exitId = CFG->getExit().getBlockID();

  // PDOM[b] = all blocks that post-dominate b
  std::map<unsigned, llvm::DenseSet<unsigned>> pdom;

  // Initialise
  for (unsigned bid : blockIds) {
    if (bid == exitId) {
      pdom[bid].insert(bid);
    } else {
      for (unsigned bid2 : blockIds)
        pdom[bid].insert(bid2);
    }
  }

  // Iterate to fixed point
  bool pdomChanged = true;
  for (int iter = 0; pdomChanged && iter < 100; ++iter) {
    pdomChanged = false;
    for (const auto *B : *CFG) {
      if (!B) continue;
      unsigned bid = B->getBlockID();
      if (bid == exitId) continue;

      llvm::DenseSet<unsigned> newPdom;
      newPdom.insert(bid);

      bool first = true;
      for (auto SI = B->succ_begin(), SE = B->succ_end(); SI != SE; ++SI) {
        const CFGBlock *Succ = *SI;
        if (!Succ) continue;
        unsigned sid = Succ->getBlockID();

        if (first) {
          newPdom.insert(pdom[sid].begin(), pdom[sid].end());
          first = false;
        } else {
          llvm::DenseSet<unsigned> intersection;
          for (unsigned x : pdom[sid])
            if (newPdom.count(x)) intersection.insert(x);
          newPdom = std::move(intersection);
        }
      }

      // A block always post-dominates itself.  The intersection above rebuilds
      // newPdom from the *successors'* sets only, so it drops the `bid` inserted
      // before the loop — without this line every block with >= 2 successors
      // (i.e. every branch predicate) silently disappears from its own pdom set.
      // That made pdom[S] == pdom[P] for the fall-through successor S of a
      // branching block, so the test below ("Y post-dominates S but not P")
      // never fired for Y == S — and Y == S is exactly the next condition block
      // of an `if/else if` chain, i.e. the chained control dependence a macro
      // like TRYCLONE introduces:
      //     if (A) {...} else if (B) {...}      ->  B's condition must be
      //                                             control-dependent on A.
      newPdom.insert(bid);

      if (newPdom != pdom[bid]) {
        pdom[bid] = std::move(newPdom);
        pdomChanged = true;
      }
    }
  }

  // Node lookup used to tag control-dependence edges with macro provenance.
  // Built here (after every Info.nodes.push_back has happened) so the
  // pointers stay valid for the rest of the function.
  std::map<std::string, const PDGNodeInfo *> nodeById;
  for (const auto &N : Info.nodes)
    nodeById[N.id] = &N;

  // Now compute control dependence: for each block P with >= 2 successors,
  // find blocks Y that post-dominate a successor of P but not P itself.
  for (const auto *B : *CFG) {
    if (!B) continue;
    unsigned bid = B->getBlockID();

    // Count valid successors
    unsigned nSucc = 0;
    for (auto SI = B->succ_begin(), SE = B->succ_end(); SI != SE; ++SI)
      if (*SI) ++nSucc;
    if (nSucc < 2) continue;

    // Find the predicate node: the block's terminator condition *is* the branch
    // condition, so resolve it through the Stmt -> node map.
    //   * ``getLastCondition()`` unwraps short-circuit chains to the operand
    //     this particular block tests (``if (a && b)`` gets one block per
    //     operand), which getTerminatorCondition() alone would not do.
    //   * The condition is the *last* statement emitted in the block — not the
    //     first.  Clang keeps the straight-line statements preceding a branch
    //     in the same block, so
    //         int r = 0;
    //         if (x > 0) { ... }
    //     holds both `int r = 0;` and the condition, and the first node of the
    //     block is the declaration.  Attributing the branch to it mislabels the
    //     predicate (and its macro provenance, which the macro-branch
    //     statistics are computed from).
    //   * A condition *variable* (``if (T x = init)``) is preferred in its
    //     declaration form, for the reason given on conditionVariableDeclStmt.
    std::string predNid;
    const Stmt *CondExpr = B->getLastCondition();
    if (!CondExpr) CondExpr = B->getTerminatorCondition();
    const Stmt *Cond = CondExpr;
    if (const DeclStmt *CondVarDS =
            conditionVariableDeclStmt(B->getTerminatorStmt()))
      Cond = CondVarDS;
    if (Cond) {
      auto Cit = stmtToNid.find(Cond);
      if (Cit != stmtToNid.end()) predNid = Cit->second;
    }
    if (predNid.empty() && CondExpr && Cond != CondExpr) {
      auto Cit = stmtToNid.find(CondExpr);
      if (Cit != stmtToNid.end()) predNid = Cit->second;
    }
    // Fallback for terminators whose condition is not itself a CFGStmt in this
    // block (macro-wrapped or recovery expressions): the condition is emitted
    // last, so take the block's last node.
    if (predNid.empty())
      for (auto Nit = Info.nodes.rbegin(); Nit != Info.nodes.rend(); ++Nit)
        if (Nit->blockId == bid) { predNid = Nit->id; break; }
    if (predNid.empty()) continue;

    for (auto SI = B->succ_begin(), SE = B->succ_end(); SI != SE; ++SI) {
      const CFGBlock *Succ = *SI;
      if (!Succ) continue;
      unsigned sid = Succ->getBlockID();

      for (unsigned yid : blockIds) {
        if (yid == bid) continue;
        bool inPdomS = pdom[sid].count(yid);
        bool inPdomB = pdom[bid].count(yid);
        if (inPdomS && !inPdomB) {
          // All nodes in block yid are control-dependent on predNid
          bool predIsMacro = false;
          auto pit = nodeById.find(predNid);
          if (pit != nodeById.end()) predIsMacro = pit->second->isMacro;
          for (const auto &N : Info.nodes)
            if (N.blockId == yid && N.id != predNid)
              Info.controlDepEdges.push_back(
                  {predNid, N.id, "if_branch", predIsMacro});
        }
      }
    }
  }

  // ---- Phase 3b: Enumerate macro-expanded branches ----
  // A "branch" here is a control-dependence predicate (a distinct
  // controlDepEdges source).  Group by predicate so each branch is counted
  // once, then keep the ones whose predicate text came from a macro.
  {
    std::map<std::string, unsigned> dependents; // predicate id -> #dependents
    for (const auto &E : Info.controlDepEdges)
      dependents[E.sourceId]++;

    for (const auto &[sid, count] : dependents) {
      auto it = nodeById.find(sid);
      if (it == nodeById.end() || !it->second->isMacro) continue;

      const PDGNodeInfo &P = *it->second;
      MacroBranchInfo MB;
      MB.nodeId = P.id;
      MB.macroName = P.macroName;
      MB.macroDefFile = P.macroDefFile;
      MB.macroDefLine = P.macroDefLine;
      MB.callSiteLine = P.sourceLine;
      MB.expression = P.expression;
      MB.dependentCount = count;
      Info.macroBranches.push_back(std::move(MB));
      Info.macroBranchByName[P.macroName]++;
    }
  }

  // ---- Phase 4: Function summary ----
  llvm::StringSet<> inputs, outputs;

  // Parameters are inputs
  for (auto *PD : FD->parameters()) {
    if (!PD) continue;
    std::string pn = PD->getNameAsString();
    if (!pn.empty()) inputs.insert(pn);
  }

  // Variables defined in nodes are outputs
  for (const auto &N : Info.nodes)
    if (!N.variable.empty()) outputs.insert(N.variable);

  // Variables used but not defined in this function → inputs
  for (const auto &due : Info.defUseEdges) {
    if (!inputs.contains(due.variable) && !outputs.contains(due.variable)) {
      bool definedHere = false;
      for (const auto &N : Info.nodes) {
        if (N.variable == due.variable) { definedHere = true; break; }
      }
      if (!definedHere) inputs.insert(due.variable);
    }
  }

  for (auto it = inputs.begin(); it != inputs.end(); ++it)
    Info.inputVariables.push_back(it->first().str());
  for (auto it = outputs.begin(); it != outputs.end(); ++it)
    Info.outputVariables.push_back(it->first().str());
}

// ===========================================================================
// AST Visitor
// ===========================================================================

class PDGBuilderVisitor : public RecursiveASTVisitor<PDGBuilderVisitor> {
public:
  explicit PDGBuilderVisitor(ASTContext &Ctx,
                              std::vector<FunctionPDGInfo> &Functions,
                              llvm::StringSet<> &SeenFunctions)
      : SM(Ctx.getSourceManager()), LO(Ctx.getLangOpts()),
        Functions(Functions), SeenFunctions(SeenFunctions) {
    GSM = &this->SM;
    GLO = &this->LO;
  }

  bool shouldVisitTemplateInstantiations() const { return true; }

  bool TraverseFunctionDecl(FunctionDecl *FD) {
    if (!FD || !FD->doesThisDeclarationHaveABody())
      return RecursiveASTVisitor::TraverseFunctionDecl(FD);
    FuncStack.push_back(FD);
    bool Ret = RecursiveASTVisitor::TraverseFunctionDecl(FD);
    FuncStack.pop_back();
    return Ret;
  }

  bool VisitFunctionDecl(FunctionDecl *FD) {
    if (!FD || !FD->doesThisDeclarationHaveABody())
      return true;

    FD = FD->getCanonicalDecl();
    std::string QName = FD->getQualifiedNameAsString();
    if (QName.empty()) return true;
    if (SeenFunctions.contains(QName)) return true;
    SeenFunctions.insert(QName);

    FunctionPDGInfo Info;
    Info.functionName = QName;
    // Use the body location for source file, not the canonical declaration
    // location. The canonical decl may point to a header file for functions
    // declared in a header but defined in a .cc file.
    {
      SourceLocation DefLoc = FD->getLocation();
      if (Stmt *Body = FD->getBody()) {
        SourceLocation BodyLoc = Body->getBeginLoc();
        if (BodyLoc.isValid())
          DefLoc = BodyLoc;
      }
      Info.sourceFile = getSF(DefLoc);
    }

    buildFunctionPDG(FD, Info);

    if (!Info.nodes.empty())
      Functions.push_back(std::move(Info));

    return true;
  }

private:
  SourceManager &SM;
  const LangOptions &LO;
  std::vector<FunctionDecl *> FuncStack;
  std::vector<FunctionPDGInfo> &Functions;
  llvm::StringSet<> &SeenFunctions;
};

// ===========================================================================
// AST Consumer
// ===========================================================================

class PDGConsumer : public ASTConsumer {
public:
  PDGConsumer(ASTContext &Ctx,
              std::vector<FunctionPDGInfo> &Functions,
              llvm::StringSet<> &SeenFunctions)
      : Visitor(Ctx, Functions, SeenFunctions) {}

  void HandleTranslationUnit(ASTContext &Ctx) override {
    Visitor.TraverseDecl(Ctx.getTranslationUnitDecl());
  }

private:
  PDGBuilderVisitor Visitor;
};

// ===========================================================================
// Frontend Action
// ===========================================================================

class PDGAction : public ASTFrontendAction {
public:
  PDGAction(std::vector<FunctionPDGInfo> &Functions,
            llvm::StringSet<> &SeenFunctions)
      : Functions(Functions), SeenFunctions(SeenFunctions) {}

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef File) override {
    return std::make_unique<PDGConsumer>(
        CI.getASTContext(), Functions, SeenFunctions);
  }

private:
  std::vector<FunctionPDGInfo> &Functions;
  llvm::StringSet<> &SeenFunctions;
};

// ===========================================================================
// Action Factory
// ===========================================================================

class PDGActionFactory : public tooling::FrontendActionFactory {
public:
  PDGActionFactory(std::vector<FunctionPDGInfo> &Functions,
                   llvm::StringSet<> &SeenFunctions)
      : Functions(Functions), SeenFunctions(SeenFunctions) {}

  std::unique_ptr<FrontendAction> create() override {
    return std::make_unique<PDGAction>(Functions, SeenFunctions);
  }

private:
  std::vector<FunctionPDGInfo> &Functions;
  llvm::StringSet<> &SeenFunctions;
};

// ===========================================================================
// JSON output
// ===========================================================================

static void writeJSON(llvm::raw_ostream &OS,
                      const std::vector<FunctionPDGInfo> &Functions,
                      const std::string &CompileDBPath) {
  llvm::json::Object Root;

  // Metadata
  llvm::json::Object Metadata;
  Metadata["compile_db"] = CompileDBPath;
  Metadata["function_count"] = (int64_t)Functions.size();
  Metadata["generator"] = "PDGBuilder";
  // 1.2: correct chained control dependence (post-dominator self-membership
  //      fix) and exact predicate selection (terminator condition instead of
  //      the branching block's first node).  Node/def-use output is unchanged
  //      from 1.1; only control_dep_edges and the predicate identity differ.
  // 1.3: expressions are no longer truncated to 117 chars + "...".  The cap
  //      (added before this file was under version control) hit exactly the
  //      machine-generated names — template instantiations and macro
  //      expansions — so distinct conditions collapsed to one identical text:
  //      five different EXPECT_EQ lines recorded the same AssertHelper(...)
  //      prefix, and the Stage-6 identity gate then reported a contradiction
  //      that does not exist.  Measured cost of removal on protobuf:
  //      +1.8% artifact size, same functions/nodes/CD/DU edges (the cap was
  //      text-only), longest expression 2125 chars, p99 337.  Anything that
  //      needs shorter text (prompts, result JSON) must cap at *render* time.
  Metadata["version"] = "1.3";
  Root["metadata"] = std::move(Metadata);

  // Functions
  llvm::json::Array FuncsArr;
  for (const auto &F : Functions) {
    llvm::json::Object FuncObj;
    FuncObj["function_name"] = F.functionName;
    FuncObj["source_file"] = F.sourceFile;

    // Nodes
    llvm::json::Array NodesArr;
    for (const auto &N : F.nodes) {
      llvm::json::Object NObj;
      NObj["id"] = N.id;
      NObj["kind"] = N.kind;
      NObj["variable"] = N.variable;
      NObj["expression"] = N.expression;
      NObj["type"] = N.type;
      NObj["source_line"] = (int64_t)N.sourceLine;
      NObj["source_col"] = (int64_t)N.sourceCol;
      NObj["block_id"] = (int64_t)N.blockId;
      NObj["is_call"] = N.isCall;
      NObj["callee"] = N.callee;
      // Macro provenance: expansion coordinates stay in source_line/source_col;
      // these fields say whether the text came from a macro and from where.
      NObj["is_macro"] = N.isMacro;
      NObj["macro_name"] = N.macroName;
      NObj["macro_def_file"] = N.macroDefFile;
      NObj["macro_def_line"] = (int64_t)N.macroDefLine;
      llvm::json::Array PArr;
      for (const auto &P : N.actualParams)
        PArr.push_back(P);
      NObj["actual_params"] = std::move(PArr);
      NodesArr.push_back(std::move(NObj));
    }
    FuncObj["nodes"] = std::move(NodesArr);

    // Def-use edges
    llvm::json::Array DUEArr;
    for (const auto &E : F.defUseEdges) {
      llvm::json::Object EObj;
      EObj["def_node_id"] = E.defNodeId;
      EObj["use_node_id"] = E.useNodeId;
      EObj["variable"] = E.variable;
      DUEArr.push_back(std::move(EObj));
    }
    FuncObj["def_use_edges"] = std::move(DUEArr);

    // Control dep edges
    llvm::json::Array CDEArr;
    for (const auto &E : F.controlDepEdges) {
      llvm::json::Object EObj;
      EObj["source_id"] = E.sourceId;
      EObj["target_id"] = E.targetId;
      EObj["kind"] = E.kind;
      EObj["is_macro_branch"] = E.isMacroBranch;
      CDEArr.push_back(std::move(EObj));
    }
    FuncObj["control_dep_edges"] = std::move(CDEArr);

    // Summary
    llvm::json::Array IArr;
    for (const auto &V : F.inputVariables) IArr.push_back(V);
    FuncObj["input_variables"] = std::move(IArr);

    llvm::json::Array OArr;
    for (const auto &V : F.outputVariables) OArr.push_back(V);
    FuncObj["output_variables"] = std::move(OArr);

    FuncObj["input_output_dep"] = llvm::json::Array{};

    // ---- Strategy 5: Completeness metrics ----
    {
      llvm::json::Object Completeness;
      unsigned recoveryCount = 0;
      for (const auto &N : F.nodes)
        if (N.kind == "recovery") ++recoveryCount;
      Completeness["recovery_node_count"] = (int64_t)recoveryCount;
      Completeness["total_node_count"] = (int64_t)F.nodes.size();
      Completeness["def_use_edge_count"] = (int64_t)F.defUseEdges.size();
      Completeness["control_dep_edge_count"] = (int64_t)F.controlDepEdges.size();

      // Score: 0.0 = completely unreliable, 1.0 = pristine
      double nodeScore = F.nodes.empty() ? 0.0 : 1.0 -
        (double)recoveryCount / (double)F.nodes.size();
      double dueScore = F.nodes.empty() ? 0.0 :
        std::min(1.0, (double)F.defUseEdges.size() /
                 (double)std::max((size_t)(F.nodes.size() / 4u), (size_t)1));
      // Manual rounding (avoid cmath dependency)
      double rawScore = nodeScore * 0.6 + dueScore * 0.4;
      Completeness["completeness_score"] = (double)(long long)(rawScore * 100.0 + 0.5) / 100.0;
      Completeness["has_def_use"] = !F.defUseEdges.empty();
      Completeness["has_control_dep"] = !F.controlDepEdges.empty();

      FuncObj["completeness"] = std::move(Completeness);
    }

    // ---- Macro-expansion statistics (per function) ----
    {
      unsigned macroNodes = 0;
      for (const auto &N : F.nodes)
        if (N.isMacro) ++macroNodes;

      // Distinct control-dependence predicates = branches of this function.
      std::set<std::string> branchSources;
      for (const auto &E : F.controlDepEdges)
        branchSources.insert(E.sourceId);

      llvm::json::Object MS;
      MS["total_node_count"] = (int64_t)F.nodes.size();
      MS["macro_node_count"] = (int64_t)macroNodes;
      MS["total_branch_count"] = (int64_t)branchSources.size();
      MS["macro_branch_count"] = (int64_t)F.macroBranches.size();

      double ratio = branchSources.empty()
                         ? 0.0
                         : (double)F.macroBranches.size() /
                               (double)branchSources.size();
      MS["macro_branch_ratio"] =
          (double)(long long)(ratio * 1000.0 + 0.5) / 1000.0;

      llvm::json::Object ByName;
      for (const auto &[name, cnt] : F.macroBranchByName)
        ByName[name.empty() ? "<unnamed>" : name] = (int64_t)cnt;
      MS["macro_branches_by_name"] = std::move(ByName);

      // Definition sites (file:line) the expanded branches came from.
      std::map<std::string, unsigned> defSites;
      for (const auto &MB : F.macroBranches) {
        std::string site = MB.macroDefFile.empty()
                               ? "<unknown>"
                               : MB.macroDefFile;
        site += ":" + std::to_string(MB.macroDefLine);
        defSites[site]++;
      }
      llvm::json::Object Sites;
      for (const auto &[site, cnt] : defSites)
        Sites[site] = (int64_t)cnt;
      MS["macro_def_sites"] = std::move(Sites);

      FuncObj["macro_stats"] = std::move(MS);

      // The explicitly-marked macro-expanded branches themselves.
      llvm::json::Array MBArr;
      for (const auto &MB : F.macroBranches) {
        llvm::json::Object MBObj;
        MBObj["node_id"] = MB.nodeId;
        MBObj["macro_name"] = MB.macroName;
        MBObj["macro_def_file"] = MB.macroDefFile;
        MBObj["macro_def_line"] = (int64_t)MB.macroDefLine;
        MBObj["call_site_line"] = (int64_t)MB.callSiteLine;
        MBObj["expression"] = MB.expression;
        MBObj["dependent_count"] = (int64_t)MB.dependentCount;
        MBArr.push_back(std::move(MBObj));
      }
      FuncObj["macro_branches"] = std::move(MBArr);
    }

    FuncsArr.push_back(std::move(FuncObj));
  }
  Root["functions"] = std::move(FuncsArr);

  // ---- Macro-expansion statistics (project-wide) ----
  {
    unsigned long long totalNodes = 0, macroNodes = 0;
    unsigned long long totalBranches = 0, macroBranchTotal = 0;
    unsigned long long funcsWithMacroBranch = 0, funcsWithMacroNode = 0;
    std::map<std::string, unsigned long long> byName, byDefSite;
    std::map<std::string, unsigned long long> byMacroFuncs; // name -> #functions

    for (const auto &F : Functions) {
      unsigned fMacroNodes = 0;
      for (const auto &N : F.nodes) {
        ++totalNodes;
        if (N.isMacro) { ++macroNodes; ++fMacroNodes; }
      }
      if (fMacroNodes > 0) ++funcsWithMacroNode;

      std::set<std::string> srcs;
      for (const auto &E : F.controlDepEdges)
        srcs.insert(E.sourceId);
      totalBranches += srcs.size();
      macroBranchTotal += F.macroBranches.size();
      if (!F.macroBranches.empty()) ++funcsWithMacroBranch;

      for (const auto &[name, cnt] : F.macroBranchByName) {
        byName[name.empty() ? "<unnamed>" : name] += cnt;
        byMacroFuncs[name.empty() ? "<unnamed>" : name] += 1;
      }
      for (const auto &MB : F.macroBranches) {
        std::string site = MB.macroDefFile.empty() ? "<unknown>"
                                                   : MB.macroDefFile;
        site += ":" + std::to_string(MB.macroDefLine);
        byDefSite[site]++;
      }
    }

    llvm::json::Object G;
    G["function_count"] = (int64_t)Functions.size();
    G["total_node_count"] = (int64_t)totalNodes;
    G["macro_node_count"] = (int64_t)macroNodes;
    G["total_branch_count"] = (int64_t)totalBranches;
    G["macro_branch_count"] = (int64_t)macroBranchTotal;
    G["function_count_with_macro_nodes"] = (int64_t)funcsWithMacroNode;
    G["function_count_with_macro_branches"] = (int64_t)funcsWithMacroBranch;

    double nodeRatio = totalNodes == 0 ? 0.0
                                       : (double)macroNodes / (double)totalNodes;
    double branchRatio = totalBranches == 0
                             ? 0.0
                             : (double)macroBranchTotal / (double)totalBranches;
    G["macro_node_ratio"] = (double)(long long)(nodeRatio * 10000.0 + 0.5) / 10000.0;
    G["macro_branch_ratio"] = (double)(long long)(branchRatio * 10000.0 + 0.5) / 10000.0;

    llvm::json::Object GB;
    for (const auto &[name, cnt] : byName) {
      llvm::json::Object E;
      E["branch_count"] = (int64_t)cnt;
      E["function_count"] = (int64_t)byMacroFuncs[name];
      GB[name] = std::move(E);
    }
    G["macro_branches_by_name"] = std::move(GB);

    llvm::json::Object GS;
    for (const auto &[site, cnt] : byDefSite)
      GS[site] = (int64_t)cnt;
    G["macro_def_sites"] = std::move(GS);

    Root["macro_stats"] = std::move(G);
  }

  llvm::json::Value V(std::move(Root));
  OS << V << "\n";
}

// ===========================================================================
// Main
// ===========================================================================

int main(int argc, const char **argv) {
  llvm::cl::opt<std::string> CompileDBPath(
      "compile-db", llvm::cl::desc("Path to compile_commands.json"),
      llvm::cl::Required);
  llvm::cl::opt<std::string> OutputPath(
      "output", llvm::cl::desc("Path to output PDG JSON"),
      llvm::cl::Required);
  llvm::cl::opt<bool> OptVerbose(
      "verbose", llvm::cl::desc("Print progress information"),
      llvm::cl::init(false));

  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Load compilation database
  std::string Error;
  auto DB = tooling::JSONCompilationDatabase::loadFromFile(
      CompileDBPath, Error, tooling::JSONCommandLineSyntax::AutoDetect);
  if (!DB) {
    llvm::errs() << "Error: " << Error << "\n";
    return 1;
  }

  // Data
  std::vector<FunctionPDGInfo> Functions;
  llvm::StringSet<> SeenFunctions;

  // Create tool
  auto AllFiles = DB->getAllFiles();
  tooling::ClangTool Tool(*DB, AllFiles);

  // Suppress warnings; filter problematic paths
  const std::vector<std::string> ProblematicPatterns = {
      "/mnt/e/Anaconda/", "/anaconda", "/conda", "/miniconda"};
  Tool.appendArgumentsAdjuster(
      [&](const tooling::CommandLineArguments &Args, StringRef) {
        tooling::CommandLineArguments Adj;
        for (const auto &Arg : Args) {
          bool Skip = false;
          for (const auto &Pat : ProblematicPatterns)
            if (Arg.find(Pat) != std::string::npos) { Skip = true; break; }
          if (!Skip) Adj.push_back(Arg);
        }
        Adj.push_back("-w");
        Adj.push_back("-Wno-error");
        return Adj;
      });

  PDGActionFactory Factory(Functions, SeenFunctions);

  if (OptVerbose)
    llvm::outs() << "Processing " << AllFiles.size()
                 << " compilation units...\n";

  int Result = Tool.run(&Factory);

  if (OptVerbose)
    llvm::outs() << "Found " << Functions.size() << " function PDGs.\n";

  // Write output
  std::error_code EC;
  llvm::raw_fd_ostream OS(OutputPath, EC);
  if (EC) {
    llvm::errs() << "Failed to open " << OutputPath << ": " << EC.message()
                 << "\n";
    return 1;
  }

  writeJSON(OS, Functions, CompileDBPath);
  OS.close();

  if (OptVerbose)
    llvm::outs() << "PDG data written to " << OutputPath << "\n";

  return Result;
}
