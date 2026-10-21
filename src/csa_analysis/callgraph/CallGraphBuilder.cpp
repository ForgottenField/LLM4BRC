//===- CallGraphBuilder.cpp -- Clang tool to build call graph JSON ---------===//
//
// Parses each translation unit in a compile_commands.json, records function
// definitions and call expressions, and outputs a JSON call graph.
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/JSONCompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

using namespace clang;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct FuncInfo {
  std::string qualifiedName;
  std::string sourceFile;
  unsigned sourceLine = 0;
  bool isMain = false;
  std::string linkage;
  bool isMethod = false;
  std::string className;
  std::string returnType;
  std::vector<std::string> paramTypes;
};

struct CallEdge {
  std::string caller;
  std::string callee;
  std::string sourceFile;
  unsigned sourceLine = 0;
  bool isIndirect = false;
};

// ---------------------------------------------------------------------------
// AST Visitor
// ---------------------------------------------------------------------------

class CallGraphVisitor : public RecursiveASTVisitor<CallGraphVisitor> {
public:
  explicit CallGraphVisitor(
      ASTContext &Ctx, std::vector<FuncInfo> &Functions,
      std::vector<CallEdge> &Calls, llvm::StringSet<> &SeenFunctions,
      std::map<std::tuple<std::string, std::string, std::string, unsigned>,
               bool> &SeenCalls)
      : SM(Ctx.getSourceManager()), Functions(Functions), Calls(Calls),
        SeenFunctions(SeenFunctions), SeenCalls(SeenCalls) {}

  bool shouldVisitTemplateInstantiations() const { return true; }

  // --- Traverse functions to maintain the call-stack ---
  bool TraverseFunctionDecl(FunctionDecl *FD) {
    if (!FD || !FD->doesThisDeclarationHaveABody())
      return RecursiveASTVisitor::TraverseFunctionDecl(FD);
    FuncStack.push_back(FD);
    bool Ret = RecursiveASTVisitor::TraverseFunctionDecl(FD);
    FuncStack.pop_back();
    return Ret;
  }

  // --- Record function definitions ---
  bool VisitFunctionDecl(FunctionDecl *FD) {
    if (!FD || !FD->doesThisDeclarationHaveABody())
      return true;

    // Use the canonical decl to avoid duplicates when a function is
    // defined in a header and seen from multiple TUs.
    FD = FD->getCanonicalDecl();

    std::string QName = FD->getQualifiedNameAsString();
    if (SeenFunctions.contains(QName))
      return true;
    SeenFunctions.insert(QName);

    FuncInfo Info;
    Info.qualifiedName = std::move(QName);
    Info.sourceFile = getSourceFile(FD->getLocation());
    Info.sourceLine = getSourceLine(FD->getLocation());
    Info.isMain = FD->isMain();

    // Determine linkage
    switch (FD->getFormalLinkage()) {
    case ExternalLinkage:
      Info.linkage = "external";
      break;
    case InternalLinkage:
      Info.linkage = "internal";
      break;
    case NoLinkage:
      Info.linkage = "none";
      break;
    default:
      Info.linkage = "other";
      break;
    }

    // Check if this is a class/struct method
    if (auto *MD = dyn_cast<CXXMethodDecl>(FD)) {
      Info.isMethod = true;
      if (auto *RD = dyn_cast<RecordDecl>(MD->getDeclContext()))
        Info.className = RD->getName().str();
    }

    // Return type
    Info.returnType = FD->getReturnType().getAsString();

    // Parameter types
    for (auto *PD : FD->parameters())
      Info.paramTypes.push_back(PD->getType().getAsString());

    Functions.push_back(std::move(Info));
    return true;
  }

  // --- Record call expressions ---
  bool VisitCallExpr(CallExpr *CE) {
    if (FuncStack.empty())
      return true;

    FunctionDecl *caller = FuncStack.back();
    FunctionDecl *callee = nullptr;

    if (auto *MCE = dyn_cast<CXXMemberCallExpr>(CE)) {
      callee = MCE->getMethodDecl();
    } else {
      callee = CE->getDirectCallee();
    }

    if (!callee || !callee->getIdentifier())
      return true; // indirect call, skip

    callee = callee->getCanonicalDecl();
    caller = caller->getCanonicalDecl();

    std::string CallerName = caller->getQualifiedNameAsString();
    std::string CalleeName = callee->getQualifiedNameAsString();

    // Skip self-recursion to reduce noise
    if (CallerName == CalleeName)
      return true;

    SourceLocation Loc = CE->getBeginLoc();
    std::string File = getSourceFile(Loc);
    unsigned Line = getSourceLine(Loc);

    auto Key = std::make_tuple(CallerName, CalleeName, File, Line);
    if (SeenCalls.count(Key))
      return true;
    SeenCalls[Key] = false;

    CallEdge Edge;
    Edge.caller = std::move(CallerName);
    Edge.callee = std::move(CalleeName);
    Edge.sourceFile = std::move(File);
    Edge.sourceLine = Line;
    Edge.isIndirect = false;
    Calls.push_back(std::move(Edge));

    return true;
  }

private:
  SourceManager &SM;
  std::vector<FunctionDecl *> FuncStack;
  std::vector<FuncInfo> &Functions;
  std::vector<CallEdge> &Calls;
  llvm::StringSet<> &SeenFunctions;
  std::map<std::tuple<std::string, std::string, std::string, unsigned>, bool>
      &SeenCalls;

  std::string getSourceFile(SourceLocation Loc) {
    if (Loc.isInvalid())
      return "";
    return SM.getFilename(Loc).str();
  }

  unsigned getSourceLine(SourceLocation Loc) {
    if (Loc.isInvalid())
      return 0;
    return SM.getSpellingLineNumber(Loc);
  }
};

// ---------------------------------------------------------------------------
// AST Consumer
// ---------------------------------------------------------------------------

class CallGraphConsumer : public ASTConsumer {
public:
  CallGraphConsumer(
      ASTContext &Ctx, std::vector<FuncInfo> &Functions,
      std::vector<CallEdge> &Calls, llvm::StringSet<> &SeenFunctions,
      std::map<std::tuple<std::string, std::string, std::string, unsigned>,
               bool> &SeenCalls)
      : Visitor(Ctx, Functions, Calls, SeenFunctions, SeenCalls) {}

  void HandleTranslationUnit(ASTContext &Ctx) override {
    Visitor.TraverseDecl(Ctx.getTranslationUnitDecl());
  }

private:
  CallGraphVisitor Visitor;
};

// ---------------------------------------------------------------------------
// Frontend Action
// ---------------------------------------------------------------------------

class CallGraphAction : public ASTFrontendAction {
public:
  CallGraphAction(
      std::vector<FuncInfo> &Functions, std::vector<CallEdge> &Calls,
      llvm::StringSet<> &SeenFunctions,
      std::map<std::tuple<std::string, std::string, std::string, unsigned>,
               bool> &SeenCalls)
      : Functions(Functions), Calls(Calls), SeenFunctions(SeenFunctions),
        SeenCalls(SeenCalls) {}

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef File) override {
    return std::make_unique<CallGraphConsumer>(
        CI.getASTContext(), Functions, Calls, SeenFunctions, SeenCalls);
  }

private:
  std::vector<FuncInfo> &Functions;
  std::vector<CallEdge> &Calls;
  llvm::StringSet<> &SeenFunctions;
  std::map<std::tuple<std::string, std::string, std::string, unsigned>, bool>
      &SeenCalls;
};

// ---------------------------------------------------------------------------
// Frontend Action Factory
// ---------------------------------------------------------------------------

class CallGraphActionFactory : public tooling::FrontendActionFactory {
public:
  CallGraphActionFactory(
      std::vector<FuncInfo> &Functions, std::vector<CallEdge> &Calls,
      llvm::StringSet<> &SeenFunctions,
      std::map<std::tuple<std::string, std::string, std::string, unsigned>,
               bool> &SeenCalls)
      : Functions(Functions), Calls(Calls), SeenFunctions(SeenFunctions),
        SeenCalls(SeenCalls) {}

  std::unique_ptr<FrontendAction> create() override {
    return std::make_unique<CallGraphAction>(Functions, Calls, SeenFunctions,
                                             SeenCalls);
  }

private:
  std::vector<FuncInfo> &Functions;
  std::vector<CallEdge> &Calls;
  llvm::StringSet<> &SeenFunctions;
  std::map<std::tuple<std::string, std::string, std::string, unsigned>, bool>
      &SeenCalls;
};

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

static void writeJSON(llvm::raw_ostream &OS,
                      const std::vector<FuncInfo> &Functions,
                      const std::vector<CallEdge> &Calls,
                      const std::string &CompileDBPath) {
  llvm::json::Object Root;

  // Metadata
  llvm::json::Object Metadata;
  Metadata["compile_db"] = CompileDBPath;
  Metadata["function_count"] = (int64_t)Functions.size();
  Metadata["call_count"] = (int64_t)Calls.size();
  Root["metadata"] = std::move(Metadata);

  // Functions array
  llvm::json::Array FuncsArr;
  for (const auto &F : Functions) {
    llvm::json::Object FuncObj;
    FuncObj["id"] = F.qualifiedName;
    FuncObj["file"] = F.sourceFile;
    FuncObj["line"] = (int64_t)F.sourceLine;
    FuncObj["linkage"] = F.linkage;
    FuncObj["is_main"] = F.isMain;
    FuncObj["return_type"] = F.returnType;

    llvm::json::Array ParamTypes;
    for (const auto &PT : F.paramTypes)
      ParamTypes.push_back(PT);
    FuncObj["param_types"] = std::move(ParamTypes);

    FuncObj["is_method"] = F.isMethod;
    FuncObj["class_name"] = F.className;
    FuncsArr.push_back(std::move(FuncObj));
  }
  Root["functions"] = std::move(FuncsArr);

  // Calls array
  llvm::json::Array CallsArr;
  for (const auto &C : Calls) {
    llvm::json::Object CallObj;
    CallObj["caller"] = C.caller;
    CallObj["callee"] = C.callee;
    CallObj["file"] = C.sourceFile;
    CallObj["line"] = (int64_t)C.sourceLine;
    CallObj["is_indirect"] = C.isIndirect;
    CallsArr.push_back(std::move(CallObj));
  }
  Root["calls"] = std::move(CallsArr);

  llvm::json::Value V(std::move(Root));
  OS << V << "\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, const char **argv) {
  llvm::cl::opt<std::string> CompileDBPath(
      "compile-db", llvm::cl::desc("Path to compile_commands.json"),
      llvm::cl::Required);
  llvm::cl::opt<std::string> OutputPath(
      "output", llvm::cl::desc("Path to output call graph JSON"),
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

  // Collect data
  std::vector<FuncInfo> Functions;
  std::vector<CallEdge> Calls;
  llvm::StringSet<> SeenFunctions;
  std::map<std::tuple<std::string, std::string, std::string, unsigned>, bool>
      SeenCalls;

  // Create the tool with source paths from the compilation database
  auto AllFiles = DB->getAllFiles();
  tooling::ClangTool Tool(*DB, AllFiles);

  // Suppress warnings for cleaner output.
  // Also filter out include paths from known-incompatible environments.
  const std::vector<std::string> ProblematicPatterns = {"/mnt/e/Anaconda/",
                                                         "/anaconda", "/conda",
                                                         "/miniconda"};
  Tool.appendArgumentsAdjuster(
      [&ProblematicPatterns](const tooling::CommandLineArguments &Args,
                              StringRef) {
        tooling::CommandLineArguments Adjusted;
        for (const auto &Arg : Args) {
          bool Skip = false;
          for (const auto &Pat : ProblematicPatterns) {
            if (Arg.find(Pat) != std::string::npos) {
              Skip = true;
              break;
            }
          }
          if (!Skip)
            Adjusted.push_back(Arg);
        }
        Adjusted.push_back("-w");
        Adjusted.push_back("-Wno-error");
        return Adjusted;
      });

  CallGraphActionFactory Factory(Functions, Calls, SeenFunctions, SeenCalls);

  if (OptVerbose)
    llvm::outs() << "Processing " << AllFiles.size()
                 << " compilation units...\n";

  int Result = Tool.run(&Factory);

  if (OptVerbose)
    llvm::outs() << "Found " << Functions.size() << " functions, "
                 << Calls.size() << " call edges.\n";

  // Write output JSON
  std::error_code EC;
  llvm::raw_fd_ostream OS(OutputPath, EC);
  if (EC) {
    llvm::errs() << "Failed to open " << OutputPath << ": " << EC.message()
                 << "\n";
    return 1;
  }

  writeJSON(OS, Functions, Calls, CompileDBPath);
  OS.close();

  if (OptVerbose)
    llvm::outs() << "Call graph written to " << OutputPath << "\n";

  return Result;
}
