# Tree-sitter backend

This package is the independent, lightweight C/C++ syntax backend. It does not
import application or report models.

## Output

`TreeSitterBackend.build_index()` returns:

- resolved source-file metadata and include names;
- function definitions with qualified names, parameters, and line ranges;
- function-scoped control, statement, and entry nodes;
- control, lexical data, and sequential-flow edges;
- call sites with syntax confidence and optional unique-name resolution.

`CodeIndex.render_evidence()` converts the graph into a compact prompt section
with explicit precision warnings.

## Execution

Use `build_index_isolated()` from application code. It runs parsing in a child
process with a timeout and returns a serializable `CodeIndex`; native parser
termination is reported as an error instead of terminating the caller.

Pipeline runs require a compilation database at
`<project-root>/compile_commands.json`. Every requested source, including a
header, must have an exact entry. The backend invokes the configured compiler
with `-E`, strips compiler line directives while retaining a source map, and
then parses the configured translation unit. This selects the real
`#if/#ifdef/#elif/#else` branch without compiling the project.

The internal graph pipeline is:

```text
configured preprocessed source
  -> detached AST
  -> basic blocks and CFG edges
  -> per-block reaching definitions
  -> PDG-Lite
```

Reaching definitions use standard block-level `IN/OUT` fixed-point sets. This
preserves multiple definitions at branch joins instead of silently connecting
mutually exclusive statements in lexical order.

The backend analyzes only the caller-provided file set. It does not scan a
repository, execute a compiler, expand includes, or claim preprocessor-aware
semantics. Pointers, macros, templates, overloads, and external APIs can make
call and data edges approximate.

## Integration boundary

`pipeline/lite_graph.py` adapts this package to the existing `FunctionPDG` /
`SDG` consumer contract and builds prompt evidence. Keep pipeline and report
types out of this package so the backend can be tested and replaced alone.
