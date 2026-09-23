# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository builds an LLM-assisted **false-positive (FP) detection and proof-of-concept (POC) generation** pipeline for Clang Static Analyzer (CSA) bug reports on C/C++ projects (aria2, folly, faiss, wslay). It combines deterministic static analysis (PDG/SDG slicing, CFG branch extraction, constraint checking) with LLM calls (default provider DeepSeek) to decide whether a reported bug path is actually reachable (**TP**), a false positive (**FP**), or undecidable (**UNKNOWN**), and when reachable generates a compilable POC.

**The classification method is documented as an ordered 10-stage pipeline in
[`docs/pipeline-stages.md`](docs/pipeline-stages.md)** — read it first; this file is the index.

The package is `llm_client`, living under `src/` (no `setup.py`/`pyproject.toml` — imports rely on `src/` being on `PYTHONPATH`).

## Commands

The package is not installed; tests import `llm_client`, so `src/` must be on the path. Use **`python3.10`** (the default `python3` is older and pytest is installed under 3.10).

```bash
# Run all tests
PYTHONPATH=src python3.10 -m pytest tests/

# Single test file / single test
PYTHONPATH=src python3.10 -m pytest tests/test_fp_analysis/test_conflict_checker.py
PYTHONPATH=src python3.10 -m pytest tests/test_factory.py::TestProviderFactory::test_create_gpt_provider

# Deterministic-stage regression (no LLM) — see docs/pipeline-stages.md §9
python3.10 tools/replay_steps1to6.py <reports...>
python3.10 tools/abc_compare.py      <reports...>
python3.10 tools/probe_report_consistency.py <reports...>
```

### Main end-to-end entry point

```bash
# Loads DEEPSEEK_API_KEY from ./.env; exits if unset
python3.10 run_path_selection.py --report <path/to/report.html>
python3.10 run_path_selection.py --report <path> --source-root ~/csa_reports/project/faiss
python3.10 run_path_selection.py --project ~/csa_reports/project/faiss   # batch: scans reports/{TP,FP}
python3.10 run_path_selection.py --scan-dir /path/to/reports --pdg pdg_faiss.json
```

There are no mode switches: conflict-driven path selection is the only selection
mechanism, and POC verification (generate + LLM audit + simulated execution +
feedback loop → TP/FP/UNKNOWN) always runs.

`run_path_selection.py` adds `src/` to `sys.path` itself (so no `PYTHONPATH` is
needed), but it still needs `.env` for `DEEPSEEK_API_KEY` — **and Python ≥ 3.9**.
On a box whose default `python3` is older, `python3` fails at import with
`TypeError: 'type' object is not subscriptable` (`list[dict[str, str]]` in
`llm_client/base.py`, evaluated at class-definition time because that module
has no `from __future__ import annotations`), which looks nothing like a Python
version problem. Use `python3.10` explicitly, as the commands above do.

### C++ native tooling

CSA, call-graph, and PDG tooling is C++ and built out-of-tree (results land in `build/` subdirs):

```bash
# Reachability checker plugin (dynamically links libclang-cpp; needs clang++-14, llvm-config-14)
bash src/llm_client/csa_analysis/checker/build.sh
# Similarly: src/llm_client/csa_analysis/callgraph/build.sh, .../pdg/build.sh
```

## Architecture

Three analysis subsystems under `src/llm_client/`, plus a top-level orchestration script.

### `run_path_selection.py` (top level) — the 10-stage pipeline

The end-to-end pipeline. **The authoritative description of the method lives in
[`docs/pipeline-stages.md`](docs/pipeline-stages.md)** — stage-by-stage inputs, outputs, verdict
exits, code locations, and the design invariants. Read that before changing any stage.
The stage index:

| # | Stage | Verdict exit |
|---|-------|--------------|
| 1 | Report parsing + project assembly (`_resolve_paths`, `parse_report`, `load_sdg`) | fail-fast on bad source root/PDG |
| 2 | Path segmentation → `SegmentInfo[]` (`cfg_feasibility.segment_path`) | — |
| 3 | PDG backward slicing → `SliceMask` (`build_slice_mask`) | — |
| 4 | Assumption feasibility → `hard_constraints` + state (`check_assumptions_feasibility`) | — |
| 5 | Branch filter: CFG extraction + 4-layer conflict + A/B/C pruning (`run_branch_filter`) | — |
| 6 | Deterministic gates, no LLM, one shared early return: **6a** report self-contradiction (`report_consistency.py`), **6b** fabricated entry state (`entry_state.py`) | **FP** |
| 7 | Semantic finite-domain gate (`semantic_fp_reason`, LLM, pre-enumeration) | **FP** |
| 8 | Path-space collection + compression (`collect_path_space`, `classify_branch_relevance`, `reduce_path_space`) | — |
| 9 | Path selection: whole-path combination enumeration (`run_path_selection`) | FP / TP / UNKNOWN |
| 10 | POC verification (`_verify_path` → `SimulationVerifier`) | TP / FP / UNKNOWN |

Stages 6 and 7 are the two "conclude FP before enumerating" gates; stage 9's four generalizations
(slice-compression / structural cause-invariance / leak consensus / exploration aggregation) are
described in the doc, as is the rule that **a single-path FP never concludes on its own**.

Results JSON is written to `output/path_selection_<report>.json`.

### `csa_analysis/` — C++ native tooling
C++ plugins (`checker/` reachability, `callgraph/`, `pdg/`) built via their `build.sh` scripts.
**These are the producers of `pdg_<project>.json` / `cfg_cache/` — keep the sources and build.sh.**

### `fp_analysis/` — core FP-detection + POC pipeline
The heart of the repo. `FPAnalyzer` (`path_analyzer.py`) orchestrates LLM-driven path selection and
POC generation. Supporting modules group by role (see `docs/pipeline-stages.md` for which stage each
serves):

- **PDG/SDG + slicing**: `pdg_models`, `pdg_loader`, `pdg_augment`, `slicer`, `slice_mask`, `path_seed_extractor`.
- **CFG + segmentation**: `cfg_models`, `cfg_parser`, `cfg_feasibility`, `cfg_cache_set`, `cfg_branch_models`, `cfg_branch_extractor`, `condition_extractor`.
- **Conflict/constraint checking**: `conflict_checker`, `conflict_models`, `conflict_learner`, `constraint_miner`, `constraint_models`, `constraint_cache`, `structural_pruning`.
- **Path space + selection**: `path_space`, `branch_relevance`, `cause_invariance`, `shortest_path`, `combination_enum`.
- **Gates**: `report_consistency` (stage 6a), `entry_state` (stage 6b), `domain_facts` (stage 7 supplement).
- **Simulation**: `simulation_verifier`.
- **Parsing/support**: `html_parser`, `source_context`, `lib_knowledge`, `models`, `prompt_templates`.

### LLM provider layer (`src/llm_client/`)
Pluggable providers: `base.py` defines the `LLMProvider` ABC + `LLMRequest`/`LLMResponse`/`LLMConfig` dataclasses; `factory.py` maps provider names (`deepseek`, `claude`, `gpt`) to implementations. `errors.py` defines the exception hierarchy (`LLMClientError`, `AuthenticationError`, `RateLimitError`, `TimeoutError`, `LLMProviderError`) — providers MUST raise these, not raw SDK errors. The FP pipeline uses the DeepSeek provider by default (`deepseek-chat`).

## Data artifacts

- `pdg_<project>.json` — SDG/PDG produced by the C++ PDG builder. **One artifact per project, and it
  must declare the version this code expects** (`Metadata["version"]` in `PDGBuilder.cpp` ==
  `pdg_models.PDG_ARTIFACT_VERSION`, currently `1.3`): a stale artifact parses fine and only its
  declaration says so — `pdg_protobuf.json` sat at 1.0 for weeks and an entire 21-report evaluation ran
  on a graph missing 42% of its control-dependency edges. `build.sh` writes exactly one binary name
  (`build/pdg_builder`); `tools/build_project_deps.py` refuses to build on version drift and deletes a
  mismatched artifact; `run_path_selection._resolve_paths` refuses to analyse one. Do not keep a `.bak`
  of an old version beside it — rebuild in place. Check the whole set with
  `python3 tools/build_project_deps.py --check-config` (prints every `pdg_*.json` as OK/STALE).
  **1.3 = expressions are stored in full** (the 117-char cap was removed; it hit template/macro names,
  so five different `EXPECT_EQ` lines recorded one identical `AssertHelper(...)` text and stage 6
  reported a contradiction that does not exist). Any consumer that needs shorter text — prompts, the
  result JSON — must cap at *render* time, never by storing a truncated artifact.
- `cfg_cache/<project>/*.json` — per-function CFG caches; auto-resolved by entry function name.
- `constraint_cache/` — mined constraint cache.
- Building these for a **new** project: see [`docs/dependency-graph-build-guide.md`](docs/dependency-graph-build-guide.md)
  (§9 = the on-demand strategy) and use `tools/build_project_deps.py --project <p> --on-demand <reports dir>`.
  The seed set comes from the reports themselves (bug host + every report event file, then a bounded
  callee closure) instead of the whole tree — protobuf: 270 MB / 76 core TUs → 21 TUs. Add the project to
  `PROJECT_SPECS` in that tool *and* to `cfg_parser._CFG_PROJECT_CONFIGS` (`--check-config` warns on drift).
- Source projects + their reports live **outside the repo** under `~/csa_reports/project/<name>/`
  (e.g. `~/csa_reports/project/faiss/reports/FP/*.html`); tests resolve them via `CSA_PROJECTS_DIR`
  and skip when absent.
- `output/` — pipeline run products only (`path_selection_*.json`, `poc_*.cpp`, logs, `intermediate/`);
  gitignored and safe to empty. Reusable harness scripts live in `tools/`.

`cfg_cache/`, `pdg_*.json`, `constraint_cache/` and `output/` are **gitignored** — they are
rebuildable artifacts, documented in `docs/pipeline-stages.md`.

## Notes

- Conflict-driven selection and POC verification have no feature flags — they are the pipeline's only behaviour. Do not add switches for them.
- **A POC that supplies the bug's precondition itself is not a trigger — and the report may refute it deterministically.** Stage 10 used to ask "can *code* make CSA's precondition true?" instead of "can a real caller reach this?", so on protobuf's `numbers.cc-SimpleAtob-{11,5}-1` (both ground-truth FP) a POC that passed `nullptr` to `SimpleAtob`'s own output parameter counted as TP — even though `ABSL_RAW_CHECK(out != nullptr, …)` at `numbers.cc:109` is unconditionally compiled and its FATAL handler calls `abort()` (`raw_logging.cc:183`). Four layers now answer it: **stage 6b** (`entry_state.detect_fabricated_entry_state`, deterministic, no LLM) fires when the report's entry assumption is "pointer value is null", the deref'd variable is a pointer parameter of the flagged function, that function's own body carries an always-compiled fatal check on it, and the report's own path records taking the check's *failure* arm before the deref — the two are incompatible, so FP before enumeration (2/21 protobuf, 0/32 on faiss+aria2+folly). The family is `ABSL_RAW_CHECK`/`ABSL_INTERNAL_CHECK`/`ABSL_CHECK`/`GOOGLE_CHECK`/`CHECK`/`FAISS_THROW_IF_NOT{,_FMT,_MSG}` — `assert`/`DCHECK`/`FAISS_ASSERT` are excluded *on purpose* (NDEBUG compiles them out, which is why CSA's null survives them) and the two-operand `CHECK_EQ` forms are skipped (the first argument cannot express "must be non-null"); a single `!V` is rejected (`!!V` is truthiness). The call-site scan (`scan_call_sites`) is **corroboration only** — it appears in the reason and can never be the sole basis of a verdict, because a public library API's callers may live outside the corpus. **F2**: `SourceContextExtractor.extract_callee_sources` now gives both the generator and the auditor the real bodies of the report's `Calling 'X'` callees (previously only the bug file was visible, so a callee in a differently-named file — `time.cc` — was invisible, which is how an "output parameter never written" claim survived the audit). **F3/F4**: the generation prompt's `// POC_UNREACHABLE: <reason>` first line is parsed by `parse_poc_unreachable_note` and injected into the simulation prompt as a **claim to adjudicate** (never a conclusion, first round only), and the audit reports `bug_value_source` — `poc_argument` **together with** `triggered` overrides the round to FP, any other value never does (null reaching a nonnull parameter from a real caller's data path is TP, per the standing calibration).
- **Do not remove the `reduced_code`-shaped assumption blind spot twice.** `analysis.reduced_code` is hard-set to `""` in `_verify_path` and never assigned, so `SimulationVerifier._source_context` always fell through to the bug file's 60-line window plus same-stem companions. That is *why* F2's callee-source block was needed; if `reduced_code` is ever wired up for real, check it does not shrink the callee evidence the auditor now receives.
- **Never substitute another project for an unresolvable one.** `_resolve_paths` used to infer the source root from the report path and, when the inferred directory had neither `deps/` nor `lib/`, silently swap in `<repo>/project/aria2`. Only aria2 has that pair, so `--report <faiss report>` loaded `pdg_aria2.json` (24,019 functions) with an empty `cfg_cache/aria2/` and produced a *plausible-looking* wrong run (18 branches, bound 32,768→1, both 484 reports "UNKNOWN") instead of an error. It now accepts the inferred root when it carries any of `deps/ lib/ src/ .git CMakeLists.txt Makefile meson.build configure setup.py`, and otherwise raises with the `--project` / `--source-root` fix in the message; a missing `pdg_<project>.json` raises too (another project's PDG parses fine and silently analyses the wrong call graph). `main()` pre-resolves every report before the first LLM call, so a misinvocation fails in seconds, not after tens of minutes as a per-report ERROR row.
- **Segments resolve to functions by (file, line range), not by name.** A `SegmentInfo` carries only a bare method name (or, inside a macro expansion, the macro's name or nothing), so name-only lookup landed on same-named functions from other classes, other TUs, or libstdc++ (`search` → `IndexPQ::search` instead of `MultiIndexQuantizer::search`; `read_index` → `read_index_header`). `CFGBranchExtractor.resolve_segment_pdg` now verifies by node lines and falls back to the segment's own file; `CFGCacheSet.find_fn_key` prefers the requested file and strips the return type's `*`. Keep any future resolution verifiable against `(file, line)` — and when a previously-failing lookup starts succeeding, gate on its *output* (`_pdg_yields`) so an "empty success" cannot silently swallow the CFG fallback.
- **A truncated LLM reply is not malformed JSON — and must never certify a TP.** Every pipeline call used to go out with `FPAnalyzer(max_tokens=4096)` (the constructor default; `run_path_selection` never overrides it, and `LLMConfig.default_max_tokens` is dead because `_call_llm` always sets `request.max_tokens`). The simulated-execution reply — the longest in the pipeline, one `path_conformance` record per report path step plus a critical-node audit — was cut off mid-string at ~12.2k chars in 5 of 8 rounds on read_index-484; the strict parse raised, `_heuristic_parse` regexed `"triggered": true` out of the incomplete text, and `SimulationResult.path_conformance_ok` then defaulted to `True`, silently disarming the hallucination gate (both 484 TPs came from such a round). Three invariants now hold: `LLMResponse.finish_reason` survives the provider layer and `_call_llm` logs a WARNING when it is `length`/`max_tokens`; the default budget is 8192 (`_SIMULATION_MAX_TOKENS` for the simulation/refine calls, DeepSeek's `deepseek-chat` cap) and `_call_llm(..., max_tokens=N)` overrides it per call; and a round whose reply had to be repaired (`_parse_json_response_tolerant`) or regex-parsed is marked `parse_degraded`, which forces `path_conformance_ok=False` so the round regenerates instead of being accepted (never flipped to FP — it degrades to UNRESOLVED). `constraint_completion` was truncated the same way and only *looked* fine because the tolerant parser salvaged a prefix.
- **The simulation's step audit is reported sparsely: `steps_checked` + `path_mismatches`, nothing per satisfied step.** The reply is the pipeline's longest and most expensive by far — on read_index-484, 3 sim calls per report at 17.1k–22.2k chars each and 15.6–21.0 s, i.e. 42–50% of the run's output tokens and ~56 s of its wall clock, while every other stage stayed under 5.8k chars. The step audit was 78% of that reply and 78% of *it* was JSON scaffolding (a 5-key record per step) with another 17% echoing back the step text the prompt had just supplied, so a conforming round paid ~15k chars for information worth ~230. `SimulationVerifier._csa_step_rows` is now the single source of step numbering (the rendered list and the reply's indices must never be numbered independently), and `_expand_sparse_audit` rebuilds the full per-step `path_conformance` records from `{"steps_checked": N, "path_mismatches": [{"step", "poc_result", "why"}]}` — everything unlisted is satisfied, so the console audit, `_format_conformance_mismatches` and the result JSON keep reading the old shape (mismatch records carry a locally-filled `report_expectation`). Measured: sim replies 17.1k–22.2k → 2.7k–6.9k chars (−77%), report output 131.7k → 97.8k chars. Two richer aggregates were tried and **removed** — each cost a decisive round over a formatting slip, never a soundness gain:
  - a one-char-per-step BIT STRING: deepseek emitted 90 bits for 92 steps while reporting `steps_checked=92`, which discarded a decisive round and flipped 484-1 TP→UNKNOWN (a homogeneous long string is a miscount trap);
  - a `steps_satisfied` COUNT: on 484-2 the model wrote `steps_satisfied: 0` next to 11 mismatch rows for 110 steps, meaning "the path as a whole is not reproduced" rather than "11 of 110 steps hold", and the arithmetic check rejected the round.
  Keep the aggregate to `steps_checked` (it must equal the rendered step count — a partial audit must not look like a pass) plus the mismatch LIST, which is the audit of record and the regeneration feedback. A `steps_satisfied` volunteered by the model is tolerated and ignored. The gate is *stronger* than the old one where it matters: `path_conformance_ok` is now derived from the list (a `False` with an empty list, a duplicate or out-of-range step index, or `steps_checked` ≠ N all mark the round `parse_degraded`, which the conformance gate already treats as "no evidence — regenerate, never TP", never an FP flip).
- **Prompt rendering is capped** (`_MAX_RENDERED_DECISIONS`, `_MAX_RENDERED_STATE_CHARS` in `path_analyzer`). Real path spaces reach tens of thousands of branches per segment, and the LLM's `established_state` can be megabytes; rendering it all overflowed the model context. Do not remove the caps — the model still gets the counts and the leading decisions.
- **Branch extraction expands only what the reported path can reach** (A/B/C in `cfg_branch_extractor`). A pure line-range scan inside a segment pulled in the *untaken* arms of every `if` in it: on `read_index`-484-2 that put 89.7% of segment 21's nodes under `read_ivf_header` at L825, inside an `IwSh` body the report demonstrably skips, and 68% of the whole tree inside libstdc++ (the `FAISS_THROW_IF_NOT` macro's `std::string` bookkeeping resolves by bare name to `std::vector::resize`). Three rules, all conservative:
  - **(A) reachability** — `_on_path_lines(fn)` walks the function's CFG from `entry_block_id` and, at every branch block with a CSA decision, follows `successors[0]` for *true* and `successors[1]` for *false*. That ordering is Clang's convention and is **verified**, not assumed: a ternary came out as `[IndexIDMap2, IndexIDMap]` = `[then, else]`. Clang also splits `a || b` into one block per operand, so a line may own several blocks — only prune when every decided line on the block agrees. If `exit_block_id` turns out unreachable, or a segment's `line_end` is not on the path, A **returns `None` and prunes nothing** (with a `logger.warning`): a wrong successor order then shows up as a sanity failure, never as silently-missing code.
    - **A rests on a pairing that does not always hold, so it verifies it.** The walk pairs CFG block `B<n>` with the PDG nodes whose `block_id == n`, but the two come from *different frontends*: the cache from `clang -cc1 -analyze -analyzer-checker=debug.DumpCFG`, which inserts implicit/temporary-dtor and `do ... while` scope blocks; the PDG from PDGBuilder's `CFG::buildCFG` with default `BuildOptions`, which does not. Every inserted block shifts all later ids — silently, without breaking any parse. On the faiss cache 984/1000 functions disagree (median shift 1–2 blocks, i.e. the EXIT/ENTRY blocks), and the large ones drift badly: for `read_index` the PDG puts `fourcc("IxFI")` (L507) in block 1161 where the cache has the L755 `snprintf` body, the ladder is really at `B1295`, and the total shift is 414. `_alignment_score` therefore compares the two numberings by *content* (Clang emits one statement per subexpression, so a block "agrees" when one of its statements is a substring of a PDG expression recorded for the same id) and A stays off for a function unless ≥ `_MIN_ALIGNED_BLOCKS` (5) blocks are comparable and ≥ `_MIN_ALIGNMENT` (0.7) of them agree. On faiss that is only the small functions — `read_index` scores 0.17, so **A is currently inert on the faiss reports and B/C carry the reduction**. Making A effective needs the block graph to come from the PDG side (emit per-function block id → source lines + successors in `pdg_<project>.json`), not from the cache; do not "fix" it by trusting the ids again.
  - **(B) slice retention** — beyond `shallow_callee_depth` (default 1), a callee is expanded only if `slice_mask.pdg_retained_nodes` kept it. 99.9% of the 484-2 root branches had an unretained callee.
  - **(C) structural guards** — an expansion stack (kills the `read_index` ↔ `read_ivf_header` mutual recursion at `index_read.cpp:443`, which a `nested_callee_name == callee_name` comparison never caught because it compared `faiss::read_index` against the short name); no recursion into callees whose PDG `source_file` is outside `source_root` (their own predicates are kept — they can still contradict a constraint — but their internals never are); and `path_space._MAX_SERIALIZED_DECISIONS` caps how much of each candidate's branch list reaches the result JSON (7.3 GB for one 484-1 run), while `signature` and the `ci_branch_keys` grouping still use the **full** list.
  - **The leaf-sentinel contract**: a callee that is *not* expanded still leaves a node with `callee_name` set, `condition_expr == "→ name()"` and `node_id == "callee_<caller>_<callee>_L<line>"`. `shortest_path` (S/C metrics), `conflict_checker` (`ReturnValueMapper`, CSA postcondition confirmation), `branch_relevance._parse_callee` and `path_space._normalize_call_edge` all read that shape — changing it silently breaks conflict checking. For the same reason a gated callee with no predicates and no calls emits *no* node (`_callee_has_content`), so gating can only ever remove nodes, never add them.
- `tools/replay_steps1to6.py` replays the deterministic stages (1–6, no LLM) for a set of reports and prints per-segment `fn / file / CFG key / branches`. Use it to compare extraction changes; it builds a **fresh `CFGCacheSet` per report** on purpose (a shared merged cache would let one report's TUs change another's answers). `tools/abc_compare.py` runs the same stages twice per report (A/B/C off vs on) and prints per-segment node/branch counts plus the whole-path bound — the regression set is 484-2, 484-1, `clone_index` (must stay 2 segments / 4 branches / bound 16), `fourcc`, `IndexFastScan`, `read_VectorTransform`, `test_merge`. See `docs/pipeline-stages.md` §9 for all four tools.
- **The design invariants that must survive any refactor are listed in `docs/pipeline-stages.md` §8** (no mode switches, `insufficient` is the conservative default, A/B/C are subtract-only, the leaf-sentinel contract, single-path FP never concludes, the sparse simulation audit, the rendering caps, the 8192 token budget, fail-fast path resolution).
- The LLM client layer (`llm_client/` top-level) is a reusable library; the `fp_analysis`/`csa_analysis` packages build on top of it.
- Tests are grouped by subsystem under `tests/` (`test_fp_analysis/`), with its own `conftest.py` and `fixtures/` for sample reports/plists.
