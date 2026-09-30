# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository builds an LLM-assisted **false-positive (FP) detection and proof-of-concept (POC) generation** pipeline for static-analyzer bug reports (Clang Static Analyzer HTML, CodeQL/Infer/Cppcheck SARIF) on C/C++ projects (aria2, folly, faiss, wslay). It combines deterministic static analysis (PDG/SDG slicing, CFG branch extraction, constraint checking) with LLM calls (default provider DeepSeek) to decide whether a reported bug path is actually reachable (**TP**), a false positive (**FP**), or undecidable (**UNKNOWN**), and when reachable generates a compilable POC.

**The classification method is documented as an ordered 10-stage pipeline in
[`docs/pipeline-stages.md`](docs/pipeline-stages.md)** — read it first; this file is the index.
**[`docs/report-frontend.md`](docs/report-frontend.md)** documents the layer stage 1 now starts
with (one report model for every analyzer + the code a report names); read it before touching
report parsing, path resolution, or the build tool's seed selection.

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

# Other report formats (see docs/report-frontend.md); these two need no API key
python3.10 run_path_selection.py --report results.sarif --report-format sarif
python3.10 run_path_selection.py --report 'results.sarif::cpp/leap-year@AlignedTable.h:66'
python3.10 run_path_selection.py --list-formats
python3.10 run_path_selection.py --list-findings --report results.sarif
```

One *finding* is one unit of work: `--scan-dir`/`--project` expand a multi-finding document
(CodeQL's SARIF) into one work item per finding, each with its own result JSON and output name
(`<stem>__<selector>`), while a CSA report keeps its bare filename so existing outputs are unchanged.

There are no mode switches: conflict-driven path selection is the only selection
mechanism, and POC verification (generate + LLM audit + simulated execution +
feedback loop → TP/FP/UNKNOWN) always runs. `--report-format` is not a mode switch — it only
pins the adapter when two formats score equally; `--list-formats`/`--list-findings` are the
self-check flags, and both exit before the API-key check.

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

Five subsystems under `src/llm_client/` (report frontend, FP pipeline — engine + application,
C++ tooling, LLM providers), plus a thin top-level runner. The ten-stage application lives in
`fp_analysis/pipeline/`; `run_path_selection.py` at the root only runs it.

### `run_path_selection.py` (top level) — the runner; `fp_analysis/pipeline/` — the 10 stages

`run_path_selection.py` is **only the runner** (~216 lines): shebang, `sys.path.insert`, the `.env`
loader, `logging.basicConfig`, the argparse block, `main()`, and a **compat shim** re-exporting the
names the harness tools and the CLI tests import from here (see below). The pipeline itself is the
subpackage **`fp_analysis/pipeline/`** — `layout` (repo-root anchor + the one logger),
`models` (the dataclasses stages hand each other), `artifacts` (result identity/path/`code_model`
block + the single `_write_output`), `project_paths` (stage 1's assembly half), `stages` (the
canonical numbering + stages 1–5 drivers), `selection` (stages 9/10 drivers + verdict prose),
`exploration` (the attempt loop + the post-loop aggregation), `runner` (`process_report` — the ten
stages as ten calls), `batch` (work-item collection + the batch driver).

**The authoritative description of the method lives in
[`docs/pipeline-stages.md`](docs/pipeline-stages.md)** — stage-by-stage inputs, outputs, verdict
exits, code locations, and the design invariants. Read that before changing any stage.
The stage index (`runner.process_report` calls them top to bottom):

| # | Stage | Adapter (in `fp_analysis/pipeline/runner.py`) | Verdict exit |
|---|-------|-----------------------------------------------|--------------|
| 1 | Report parsing + project assembly (`_resolve_paths`, `_infer_source_root`, `_infer_entry_function`, `parse_report`, `load_sdg`; parsing itself is `ReportFrontend` → `report_bridge`) | `_stage1_parse` / `_stage1_assemble` / `_stage1_resolve_entry` | fail-fast on bad source root/PDG |
| 2 | Path segmentation → `SegmentInfo[]` (`cfg_feasibility.segment_path`; **skipped for a location-only report** — `ReportMetadata.path_evidence`) | `_stage2_segment` / `_stage2_extend_caches` | — |
| 3 | PDG backward slicing → `SliceMask` (`build_slice_mask`) | `_stage3_slice` | — |
| 4 | Assumption feasibility → `hard_constraints` + state (`check_assumptions_feasibility`) | `_stage4_assumptions` | — |
| 5 | Branch filter: CFG extraction + 4-layer conflict + A/B/C pruning (`run_branch_filter`) | `_stage5_branch_filter` | — |
| 6 | Deterministic gates, no LLM, one shared early return: **6a** report self-contradiction (`report_consistency.py`), **6b** fabricated entry state (`entry_state.py`) | `_stage6_gate` | **FP** |
| 7 | Semantic finite-domain gate (`semantic_fp_reason`, LLM, pre-enumeration) | `_stage7_preamble` / `_stage7_supplement` / `_stage7_gate` | **FP** |
| 8 | Path-space collection + compression (`collect_path_space`, `classify_branch_relevance`, `reduce_path_space`) | `_stage8_path_space` | — |
| 9 | Path selection: whole-path combination enumeration (`run_path_selection`) | `exploration._run_path_attempts` | FP / TP / UNKNOWN |
| 10 | POC verification (`_verify_path` → `SimulationVerifier`; **skipped for a location-only report** → UNKNOWN, `_no_path_verdict`) | `exploration._run_path_attempts` (+ `_aggregate_exploration`) | TP / FP / UNKNOWN |

Stages 6 and 7 are the two "conclude FP before enumerating" gates; stage 9's four generalizations
(slice-compression / structural cause-invariance / leak consensus / exploration aggregation) are
described in the doc, as is the rule that **a single-path FP never concludes on its own**.

**The stage bodies are verbatim.** The split preserved the code, not just the behaviour: 1092 of the
old function's 1124 body lines are byte-identical, and the 32 that are not are the deliberate edits
(the dead `verdict_steps`/`poc_*` locals, three copies of the write-plus-two-prints trio collapsed
into `artifacts._write_output`, the state initializers moved onto the two state dataclasses, and the
`report_path`→`report_ref.path` banner line). Two consequences to keep: **`t0` is a parameter**, not
a fresh clock — the old body used one local for stage 3's elapsed print *and* the analyzer's print,
then re-set it for stage 5 and its gate (see `runner`'s module docstring), so a callee that mints its
own `t0` changes the printed durations; and **the loop's body is verbatim too**, with its inputs
aliased to the state carriers at the top and written back once at the bottom
(`exploration._run_path_attempts`) — do not "simplify" that into attribute reads and writes.

**The compat shim in the script is a contract, not leftovers.** `tools/{replay_steps1to6,
abc_compare,probe_report_consistency}.py` and both CLI test files import 20 names from
`run_path_selection`; `tests/test_fp_analysis/test_pipeline_shim.py` derives that list from the
consumers' own source and asserts each alias is the *same object* as the pipeline module's
attribute, so the block cannot be deleted as dead code. Note the flip side when patching in a test:
**a patch must target the module that makes the call.** `process_report`'s `_resolve_paths` and
`parse_report` are looked up in `pipeline.runner`'s globals; the attempt loop's `run_path_selection`
and `_verify_path` in `pipeline.exploration`'s. A re-export does not help — it binds a second,
unread name.

Results JSON is written to `output/path_selection_<report>.json`.

### `report_frontend/` — the report frontend (stage 1's first half)
One canonical model for **every** static analyzer's report, plus the code the report names.
**Full description: [`docs/report-frontend.md`](docs/report-frontend.md)**; the modules are:

- **Canonical report model**: `models` (`ReportDocument`, `BugFinding`, `TraceStep`, `StepKind`,
  `SourceLocation`, `Severity`, `AnalyzerKind`, `ReportFormat`).
- **Canonical code model**: `code_model` (`SourceCodeModel`, `ResolvedFile`) — one resolution
  ladder (`direct`/`build-prefix`/`relative`/`suffix`/`basename`/`embedded`/`unresolved`) with the
  matched strategy recorded, `STRONG_STRATEGIES` excluding the name-only rungs, and snippets read
  from the report's embedded copy first. **A file the document lists gets its own per-section copy
  and only that** (`frontend._register_embedded`): a CSA report's `raw["source_snippets"]` is a
  whole-document window filtered by line number alone, so it also carries other sections' lines of
  that number — or, where the bug file is blank, plain fills in a neighbouring file's code
  (1828 lines / 62 reports measured; 25 had the *bug line itself* foreign). Merging the two copies
  at one key cannot fix it: `update` overwrites a wrong line but never removes one. `snippet()`
  also records **drift** when the report's copy and the local tree disagree at the same line
  (`{reported, line, embedded, local, strategy}`, deduped, capped, `drift_truncated` counts what the
  cap dropped) → `logger.warning` + the result JSON's `code_model` key. It is **visibility only**:
  it never participates in a verdict and is not a switch.
- **Adapters**: `adapters/csa_html` (CSA HTML), `adapters/sarif` (CodeQL / Infer / Cppcheck);
  `registry` (+`adapters/base` for `--list-formats`) does content sniffing, ambiguity handling and
  the `--report-format` override.
- **Facade**: `frontend` (`ReportFrontend.load`/`expand`/`iter_findings`, `LoadedReport`,
  `ReportRef` = `<path>::<finding>`, `os.PathLike` so existing `Path(...)`/`.stem` call sites keep working).

**The pipeline still reads `ParsedReport`/`BugPath`**: `fp_analysis/report_bridge` renders the
canonical finding back into the legacy dicts the 26k lines below stage 1 expect — losslessly for CSA
(golden-tested), and *without inventing* the fields a non-CSA format does not have (a SARIF finding
has no error-start step, no `Calling 'X'` wording, no branch direction), so the stages that look for
them stay conservative.

### `csa_analysis/` — C++ native tooling
C++ plugins (`checker/` reachability, `callgraph/`, `pdg/`) built via their `build.sh` scripts.
**These are the producers of `pdg_<project>.json` / `cfg_cache/` — keep the sources and build.sh.**

### `fp_analysis/` — core FP-detection + POC pipeline
The heart of the repo. `FPAnalyzer` (`path_analyzer.py`) orchestrates LLM-driven path selection and
POC generation. Supporting modules group by role (see `docs/pipeline-stages.md` for which stage each
serves):

- **`pipeline/`** — the **application** layer (subpackage + facade, same shape as `report_frontend/`):
  `layout`, `models`, `artifacts`, `project_paths`, `stages`, `selection`, `exploration`, `runner`,
  `batch`. It is the only code that knows the CLI's arguments, the console's output and the result
  JSON's shape; it never imports the entry script. See the section below and
  `docs/pipeline-stages.md` §1.

- **PDG/SDG + slicing**: `pdg_models`, `pdg_loader`, `pdg_augment`, `slicer`, `slice_mask`, `path_seed_extractor`.
- **CFG + segmentation**: `cfg_models`, `cfg_parser`, `cfg_feasibility`, `cfg_cache_set`, `cfg_branch_models`, `cfg_branch_extractor`, `condition_extractor`.
- **Conflict/constraint checking**: `conflict_checker`, `conflict_models`, `conflict_learner`, `constraint_miner`, `constraint_models`, `constraint_cache`, `structural_pruning`.
- **Path space + selection**: `path_space`, `branch_relevance`, `cause_invariance`, `shortest_path`, `combination_enum`.
- **Gates**: `report_consistency` (stage 6a), `entry_state` (stage 6b), `domain_facts` (stage 7 supplement).
- **Simulation**: `simulation_verifier`.
- **Parsing/support**: `source_context`, `lib_knowledge`, `models`, `prompt_templates` — plus
  `html_parser`, which is now only a **compatibility shim** over `report_frontend` + `report_bridge`
  (its parsing moved to `report_frontend/adapters/csa_html`). New code should use the frontend
  directly; the shim exists so the existing call sites and tests did not have to churn in the same
  change as the formats refactor. `report_bridge` is the *pipeline-wide* seam of the same kind.

### LLM provider layer (`src/llm_client/`)
Pluggable providers: `base.py` defines the `LLMProvider` ABC + `LLMRequest`/`LLMResponse`/`LLMConfig` dataclasses; `factory.py` maps provider names (`deepseek`, `claude`, `gpt`) to implementations. `errors.py` defines the exception hierarchy (`LLMClientError`, `AuthenticationError`, `RateLimitError`, `TimeoutError`, `LLMProviderError`) — providers MUST raise these, not raw SDK errors. The FP pipeline uses the DeepSeek provider by default (`deepseek-chat`).

## Data artifacts

- `pdg_<project>.json` — SDG/PDG produced by the C++ PDG builder. **One artifact per project, and it
  must declare the version this code expects** (`Metadata["version"]` in `PDGBuilder.cpp` ==
  `pdg_models.PDG_ARTIFACT_VERSION`, currently `1.3`): a stale artifact parses fine and only its
  declaration says so — `pdg_protobuf.json` sat at 1.0 for weeks and an entire 21-report evaluation ran
  on a graph missing 42% of its control-dependency edges. `build.sh` writes exactly one binary name
  (`build/pdg_builder`); `tools/build_project_deps.py` refuses to build on version drift and deletes a
  mismatched artifact; `pipeline.project_paths._resolve_paths` refuses to analyse one. Do not keep a `.bak`
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
  It parses through the report frontend now (`--on-demand <dir>` globs `report-*.html` **and** `*.sarif`)
  and resolves paths through `SourceCodeModel` — it has no ladder of its own, on purpose: the one it had
  lacked the `relative` rung, so a SARIF's relative `uri` could only be placed by basename. Its weak-match
  rule is graded — a **bug file** placed by name alone is a GAP (that is the report's own starting TU; the
  message lists every file sharing the name), an ambiguous **event file** is kept (it only widens the build).
  After any change here, re-run the seed comparison in `docs/report-frontend.md` §9 (53 CSA reports across
  faiss/folly/aria2/protobuf: expect 0 changed, identical TU sets).
- Source projects + their reports live **outside the repo** under `~/csa_reports/project/<name>/`
  (e.g. `~/csa_reports/project/faiss/reports/FP/*.html`); tests resolve them via `CSA_PROJECTS_DIR`
  and skip when absent.
- `output/` — pipeline run products only (`path_selection_*.json`, `poc_*.cpp`, logs, `intermediate/`);
  gitignored and safe to empty. Reusable harness scripts live in `tools/`.

`cfg_cache/`, `pdg_*.json`, `constraint_cache/` and `output/` are **gitignored** — they are
rebuildable artifacts, documented in `docs/pipeline-stages.md`.

## Notes

- Conflict-driven selection and POC verification have no feature flags — they are the pipeline's only behaviour. Do not add switches for them.
- **Report parsing goes through `report_frontend`; never convert one analyzer's report into another's shape.** The CodeQL findings were first integrated by hand-rewriting them as CSA HTML, which is silent and lossy in exactly the direction that manufactures FPs: the rewritten "CSA report" has no branch direction either, but a hand-edit can *supply* one, and then stage 5 sees a condition it cannot check and stage 6a a direction it was never told. The frontend's rule is the opposite — the canonical model holds only what the analyzer said, and `report_bridge` renders it into the legacy `ParsedReport`/`BugPath` **without inventing** the fields a non-CSA format lacks (no error-start step, no `Calling 'X'` callee, no `branch_condition`), so the stages that look for those find nothing and stay conservative. Two concrete defects this refactor had to fix, both silent: (a) `cfg_feasibility.build_bug_report_path` recognised the report step only by CSA's `path_number == 9999`, so a SARIF finding segmented into **zero** segments — the bridge now writes `9999` for a `StepKind.REPORT` step and the stage also accepts `event_kind == "report"` (for CSA the two always agree, so the second can only confirm the first); (b) the SARIF adapter appended a REPORT step whenever `steps[-1].location != finding_location` — a structural comparison, while a `threadFlow` location routinely pins only `startLine` and the result's location also carries `column`/`endLine`, producing `[5, 9, 12, 12]` with an empty-message REPORT; it compares by file basename + line (`_same_place`) now. **CSA output must stay byte-identical** — `tests/test_report_frontend/test_csa_fidelity.py` holds a golden captured from `git show HEAD:` plus a corpus invariant check over 211 real reports.
- **The code-resolution ladder exists once, in `report_frontend.code_model` (`SourceCodeModel`); every consumer delegates, and the ones that treat an answer as authoritative accept only `STRONG_STRATEGIES`.** `ResolvedFile` records which rung matched; `STRONG_STRATEGIES` excludes `basename`/`embedded`/`unresolved` on purpose, and `basename` is the only rung that can be ambiguous — `candidates` carries every same-named file *with the chosen one first*, so anything reporting an ambiguity must use `rivals` (the candidates minus the pick), or the message reads `X → X (also: X, Y)`, which says the opposite of what it means. Consumers: `cfg_feasibility._resolve_local_path` (via `shared_model`), `cfg_cache_set`, `run_path_selection._resolve_source_file_for_cfg`, `tools/build_project_deps.resolve_reported`. **Two of these were found by running real reports, not by reading code** — the whole set went green on 211 CSA reports while both were broken, because a CSA report spells its paths the way the weak implementations happened to handle:
  - `tools/build_project_deps.py` had its own ladder and it lacked the `relative` rung, which is the only spelling a SARIF/Cppcheck `uri` has: a CodeQL corpus could therefore only be seeded by basename guess. It delegates now; its seed set is unchanged on 53 CSA reports (faiss/folly/aria2/protobuf) and its weak-match rule is graded — a **bug file** placed by name alone becomes a GAP that lists every file sharing the name (it is the report's own starting TU), while an ambiguous **event file** is kept, since it only widens the build.
  - `cfg_feasibility._resolve_local_path` walked suffixes only (drop leading components until one exists), which can place an absolute build path but **cannot place a relative one** — there is no leading component to drop. `src/fmt.cc` (a SARIF `uri`) tried `<root>/fmt.cc` and gave up while the file is at `<root>/src/fmt.cc`, so on the real CodeQL corpus *every* event of a finding fell out of `event_files` and into `_resolve_event_file`'s line-number guess: the 8-step `aria2.sarif::cpp/non-constant-format@paramed_string.h:145` attributed its events to `src/AbstractAuthResolver.cc` / libstdc++ headers and produced one segment pointing at `basic_string.tcc` instead of the report's own `expandLoop`. It delegates now (96% of path-bearing CodeQL findings get a named function), and it **refuses weak rungs**: a name-only match is a different claim (`/usr/include/fmt/core.h` is not `<root>/lib/cpluff/.../core.h`) and its caller would treat it as authoritative, whereas `None` routes the event to the line-index fallback, which is restricted to the files the report mentions. A wrong file is worse than no file. Resolution must stay *identical* on CSA: the corpus diff (424 paths over 211 reports) is 379 same / 45 both-`None` / **0 changed**.
- **"Does this report have a path?" is a fact in the model, not a count of the steps.** It is the largest single difference between the analyzers: CSA walks a symbolic path and prints every step, while a CodeQL query often reports only *where* — **127 of the 176 findings in the real corpus have no `codeFlow` at all**, and `folly.sarif` is 5/5 location-only. A location-only finding still renders one step (the report step, `path_number == 9999`), so nothing downstream can tell it from a one-step path by counting — and the stage that must know is stage 2. So `report_frontend.models.PathEvidence` (`TRACE` | `LOCATION_ONLY`) is set by the **adapter** (which may read the whole document — that is where the analyzer's statement lives), defaulted to the weaker reading, and carried to the pipeline as `ReportMetadata.path_evidence` by `report_bridge`. **Consumers must not re-derive it from `len(steps)`.** Stage 2 branches on it: for `LOCATION_ONLY` the only boundary is the report step, so the reported location's enclosing function is the analysis unit and the path walk does not run. Both readings coincide on today's corpus (a lone report step was already the only boundary) — the rule is stated so that a *path-shaped rendering* cannot be segmented as a path by accident, which is the failure the distinction exists to prevent. A contradiction (model says no path, events carry control boundaries) logs a warning and keeps the model's reading; the model is authoritative, and a silent choice either way would be worse. Distribution over the CodeQL corpus is 49 `trace` / 127 `location`, matching the raw `codeFlow` histogram independently; all 211 CSA reports are `TRACE`, so the branch never fires for CSA. **Stage 2 is not the only consumer — stage 10 is the other, and its rule is a verdict rule.** For `LOCATION_ONLY` the pipeline does not run POC verification at all and returns **UNKNOWN** (`_no_path_verdict`): stage 9 can always *construct* a feasible path, and the instrument that turns that candidate into a TP is the step-by-step conformance audit against the **reported** path — with no reported path only the report step is left, so the audit passes vacuously and `triggered` would stand on a path the pipeline chose rather than the one the analyzer claimed (the "POC manufactures its own precondition" failure at report scale, and the audit that would have caught it is exactly what is missing). The asymmetry is deliberate: **FP says reachability is refuted and needs no path of its own; TP is a claim *about* a path** — so a location-only report may still conclude FP from stages 6/7 or stage 9's conflict-based exit, but can never conclude TP. Its `verdict_step` keyword is `报告未给出路径`, kept distinct from `遍历超限` (that one means the exploration budget ran out).
- **One finding is one unit of work.** A CSA report is one file/one finding, so "which report" and "which finding" were one question; a SARIF export separates them. `ReportRef` (`<path>::<selector>`) is `os.PathLike` so the existing `Path(...)`/`.stem`/`.name` call sites keep working, and `key` names the outputs: the bare filename when there is no selector (**existing CSA result JSON and output names are unchanged**) and `<stem>__<sanitised selector>` otherwise, so 62 findings do not overwrite each other. `expand` turns a multi-finding document into one work item per finding; `load` on a multi-finding document *without* a selector raises rather than picking one. **The selector has to survive all the way into stage 1, and it did not.** `process_report` rebinds its argument (`report_path = report_ref.path`) so the banner can print the bare path, then handed *that* to `parse_report` — so the frontend received a selector-less ref for a multi-finding document and correctly refused it ("holds 62 findings; select one"). Every run that had a selector therefore died before stage 1: `--report '<doc>::<finding>'` and the whole `--scan-dir` expansion. Collecting the right work item is not the same as analysing it; `tests/test_report_frontend/test_cli_integration.py::TestTheSelectorReachesStageOne` asserts the ref, not the path, reaches `parse_report` (verified to fail against the old line).
- **A POC that supplies the bug's precondition itself is not a trigger — and the report may refute it deterministically.** Stage 10 used to ask "can *code* make CSA's precondition true?" instead of "can a real caller reach this?", so on protobuf's `numbers.cc-SimpleAtob-{11,5}-1` (both ground-truth FP) a POC that passed `nullptr` to `SimpleAtob`'s own output parameter counted as TP — even though `ABSL_RAW_CHECK(out != nullptr, …)` at `numbers.cc:109` is unconditionally compiled and its FATAL handler calls `abort()` (`raw_logging.cc:183`). Four layers now answer it: **stage 6b** (`entry_state.detect_fabricated_entry_state`, deterministic, no LLM) fires when the report's entry assumption is "pointer value is null", the deref'd variable is a pointer parameter of the flagged function, that function's own body carries an always-compiled fatal check on it, and the report's own path records taking the check's *failure* arm before the deref — the two are incompatible, so FP before enumeration (2/21 protobuf, 0/32 on faiss+aria2+folly). The family is `ABSL_RAW_CHECK`/`ABSL_INTERNAL_CHECK`/`ABSL_CHECK`/`GOOGLE_CHECK`/`CHECK`/`FAISS_THROW_IF_NOT{,_FMT,_MSG}` — `assert`/`DCHECK`/`FAISS_ASSERT` are excluded *on purpose* (NDEBUG compiles them out, which is why CSA's null survives them) and the two-operand `CHECK_EQ` forms are skipped (the first argument cannot express "must be non-null"); a single `!V` is rejected (`!!V` is truthiness). The call-site scan (`scan_call_sites`) is **corroboration only** — it appears in the reason and can never be the sole basis of a verdict, because a public library API's callers may live outside the corpus. **F2**: `SourceContextExtractor.extract_callee_sources` now gives both the generator and the auditor the real bodies of the report's `Calling 'X'` callees (previously only the bug file was visible, so a callee in a differently-named file — `time.cc` — was invisible, which is how an "output parameter never written" claim survived the audit). **F3/F4**: the generation prompt's `// POC_UNREACHABLE: <reason>` first line is parsed by `parse_poc_unreachable_note` and injected into the simulation prompt as a **claim to adjudicate** (never a conclusion, first round only), and the audit reports `bug_value_source` — `poc_argument` **together with** `triggered` overrides the round to FP, any other value never does (null reaching a nonnull parameter from a real caller's data path is TP, per the standing calibration).
- **Do not remove the `reduced_code`-shaped assumption blind spot twice.** `analysis.reduced_code` is hard-set to `""` in `_verify_path` and never assigned, so `SimulationVerifier._source_context` always fell through to the bug file's 60-line window plus same-stem companions. That is *why* F2's callee-source block was needed; if `reduced_code` is ever wired up for real, check it does not shrink the callee evidence the auditor now receives.
- **Never substitute another project for an unresolvable one.** `_resolve_paths` used to infer the source root from the report path and, when the inferred directory had neither `deps/` nor `lib/`, silently swap in `<repo>/project/aria2`. Only aria2 has that pair, so `--report <faiss report>` loaded `pdg_aria2.json` (24,019 functions) with an empty `cfg_cache/aria2/` and produced a *plausible-looking* wrong run (18 branches, bound 32,768→1, both 484 reports "UNKNOWN") instead of an error. It now accepts the inferred root when it carries any of `deps/ lib/ src/ .git CMakeLists.txt Makefile meson.build configure setup.py`, and otherwise raises with the `--project` / `--source-root` fix in the message; a missing `pdg_<project>.json` raises too (another project's PDG parses fine and silently analyses the wrong call graph). `main()` pre-resolves every report before the first LLM call, so a misinvocation fails in seconds, not after tens of minutes as a per-report ERROR row.
- **Segments resolve to functions by (file, line range), not by name.** A `SegmentInfo` carries only a bare method name (or, inside a macro expansion, the macro's name or nothing), so name-only lookup landed on same-named functions from other classes, other TUs, or libstdc++ (`search` → `IndexPQ::search` instead of `MultiIndexQuantizer::search`; `read_index` → `read_index_header`). `CFGBranchExtractor.resolve_segment_pdg` now verifies by node lines and falls back to the segment's own file; `CFGCacheSet.find_fn_key` prefers the requested file and strips the return type's `*`. Keep any future resolution verifiable against `(file, line)` — and when a previously-failing lookup starts succeeding, gate on its *output* (`_pdg_yields`) so an "empty success" cannot silently swallow the CFG fallback.
- **A truncated LLM reply is not malformed JSON — and must never certify a TP.** Every pipeline call used to go out with `FPAnalyzer(max_tokens=4096)` (the constructor default; `pipeline.runner._build_analyzer` never overrides it, and `LLMConfig.default_max_tokens` is dead because `_call_llm` always sets `request.max_tokens`). The simulated-execution reply — the longest in the pipeline, one `path_conformance` record per report path step plus a critical-node audit — was cut off mid-string at ~12.2k chars in 5 of 8 rounds on read_index-484; the strict parse raised, `_heuristic_parse` regexed `"triggered": true` out of the incomplete text, and `SimulationResult.path_conformance_ok` then defaulted to `True`, silently disarming the hallucination gate (both 484 TPs came from such a round). Three invariants now hold: `LLMResponse.finish_reason` survives the provider layer and `_call_llm` logs a WARNING when it is `length`/`max_tokens`; the default budget is 8192 (`_SIMULATION_MAX_TOKENS` for the simulation/refine calls, DeepSeek's `deepseek-chat` cap) and `_call_llm(..., max_tokens=N)` overrides it per call; and a round whose reply had to be repaired (`_parse_json_response_tolerant`) or regex-parsed is marked `parse_degraded`, which forces `path_conformance_ok=False` so the round regenerates instead of being accepted (never flipped to FP — it degrades to UNRESOLVED). `constraint_completion` was truncated the same way and only *looked* fine because the tolerant parser salvaged a prefix.
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
