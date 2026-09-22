"""LLM prompt templates for FP path analysis, completion, POC generation, and compile fixing."""

# ---------------------------------------------------------------------------
# Bug-Type-Specific Error Start / Data Flow descriptions
# ---------------------------------------------------------------------------

ERROR_START_LEGEND_MEMORY_LEAK = (
    "⚠ = Error Start event (allocation site — where malloc/calloc "
    "allocated the memory that could be leaked)"
)

ERROR_START_LEGEND_DEFAULT = (
    "⚠ = Error Start event (value originates here) — if present in the path"
)

DATA_FLOW_SECTION_MEMORY_LEAK = (
    "## CRITICAL: Memory Allocation Flow Analysis\n"
    "\n"
    "For **Memory Leak** reports, the ⚠ Error Start marks the **allocation site** "
    "(malloc/calloc/etc.), NOT an undefined/garbage value. "
    "You MUST trace the **allocated memory** from allocation to the potential leak:\n"
    "\n"
    "1. **Identify the allocation**: Look for the ⚠ Error Start event. "
    "This is where malloc/calloc allocates the memory that may be leaked.\n"
    "2. **Trace the pointer**: Follow the allocated pointer through assignments, "
    "function parameters, return values, and data structure insertions.\n"
    "3. **Find every error path after allocation**: For each opaque function call "
    "after the allocation, determine if it can fail (check "
    "``return_value_constraints.failure_nature``). If it can fail, check whether "
    "the allocated memory is freed before the function returns on the error path.\n"
    "4. **Use the provided function summaries**: The ``alloc_sources``, "
    "``free_sinks``, ``ownership_transfer``, and ``error_path_behavior`` fields "
    "tell you which functions allocate, which free, and who owns what. "
    "Use these as facts, not guesses.\n"
    "5. **Determine the TRUE root cause**: The root cause is a missing ``free`` "
    "on an error path — an opaque function call that can fail between the "
    "allocation and the intended free site, with no cleanup.\n"
    "6. **If no explicit \"Memory is allocated\" event follows Error Start**: "
    "Infer the allocation from context — look for malloc/calloc calls near the "
    "Error Start location in the source code.\n"
    "\n"
    'The key question: "Is the allocated memory freed on EVERY execution path '
    'after the allocation, including error returns from opaque function calls?" '
    "If not, the report is a TRUE POSITIVE."
)

DATA_FLOW_SECTION_DEFAULT = (
    "## CRITICAL: Data Flow Root Cause Analysis\n"
    "\n"
    "Before listing gaps, you MUST trace the data flow from the **origin of "
    "the undefined/garbage value** to the final bug location:\n"
    "\n"
    "1. **Identify where the undefined value originates**: Look for the \"Error "
    "Start\" event (marked with ⚠ in the path). At this point, CSA marks a "
    "variable as having an undefined/garbage value.\n"
    "2. **Trace propagation**: Follow how this value is passed through function "
    "calls, assignments, and transformations. Each \"Calling '...'\" event may "
    "pass the value as an argument or receive it as a return value.\n"
    "3. **Identify opaque functions**: For each function called along the data "
    "flow path, determine whether CSA models its body. If a function is opaque "
    "(not modeled), CSA cannot see its side effects — including buffer writes, "
    "pointer initializations, or value assignments.\n"
    "4. **Use the provided function summaries**: The prompt includes \"Function "
    "Summaries\" — structured descriptions of what opaque functions actually do. "
    "Use these as facts, not guesses. For example, if a summary says a function "
    "\"writes to the output buffer unconditionally\", treat that as ground truth.\n"
    "5. **Determine the TRUE root cause**: The root cause is typically the FIRST "
    "opaque function along the data flow path that would have initialized or "
    "constrained the value if CSA had modeled its body.\n"
    "6. **If no explicit \"Error Start\" event exists in the path**: Infer the "
    "most likely origin point of the undefined value by analyzing the path "
    "events and source code. Look for:\n"
    "   - Variables declared without initialization\n"
    "   - Return values from opaque functions that are assumed to be garbage\n"
    "   - Memory that is allocated but not explicitly initialized before use\n"
    "   - The earliest point in the path where the value could become undefined\n"
    "\n"
    'The key question: "Where does the garbage/undefined value actually come '
    'from, and which opaque function\'s unmodeled side effect is the root cause?"'
)

DATA_FLOW_PHASE1_MEMORY_LEAK = (
    "### Phase 1: Memory Allocation Flow Analysis (MANDATORY)\n"
    "Before listing gaps, perform this analysis:\n"
    "\n"
    "1. **Trace the allocated memory**: Identify the pointer returned by the "
    "allocation at the Error Start event. This is the memory that could be leaked.\n"
    "2. **Find the intended free site**: Where is this memory supposed to be "
    "freed? Is there a ``free`` call in the function that owns it? Check "
    "``free_sinks`` in the relevant function summaries.\n"
    "3. **Identify opaque function calls between alloc and free**: For each "
    "opaque function called after the allocation that could fail (non-deterministic "
    "failure), check whether the allocated memory is freed before the return. "
    "If not, this is the leak path.\n"
    "4. **Check ownership semantics**: Does any opaque function take ownership "
    "of the pointer (check ``ownership_transfer`` in summaries)? If so, does it "
    "properly free on its error paths?\n"
    "5. **If no explicit \"Memory is allocated\" event**: Look for malloc/calloc "
    "calls near the Error Start location.\n"
    "\n"
    'The key question: "Is there an error path between allocation and the '
    'intended free that bypasses the free?" If yes, this is a TRUE POSITIVE '
    "with 0 gaps."
)

DATA_FLOW_PHASE1_DEFAULT = (
    "### Phase 1: Data Flow Root Cause Analysis (MANDATORY)\n"
    "Before listing gaps, perform this analysis:\n"
    "\n"
    "1. **Trace the undefined value**: Identify the variable/expression that "
    "holds the undefined/garbage value that leads to the bug report.\n"
    "2. **Find its origin**: Where was this value first marked as undefined by "
    "CSA? This is typically:\n"
    "   - An uninitialized local variable\n"
    "   - The return value of an opaque function call\n"
    "   - Memory whose contents CSA assumes are garbage\n"
    "3. **Follow every propagation step**: How does this value flow through "
    "function calls, assignments, and branches to reach the final bug location?\n"
    "4. **Check each opaque function**: For every opaque function on the data "
    "flow path, determine whether it would have initialized or constrained the "
    "value if CSA had modeled its body. The TRUE root cause is typically the "
    "earliest opaque function along the data flow path whose side effect CSA missed.\n"
    "5. **If the report has no explicit \"Error Start\" event**: Infer the most "
    "likely origin point from context — look at variable declarations, "
    "uninitialized memory, and opaque function return values near the beginning "
    "of the path."
)

# ---------------------------------------------------------------------------
# Prompt 1: Path Gap Analysis
# ---------------------------------------------------------------------------

GAP_ANALYSIS_SYSTEM_PROMPT = """You are a C++ static analysis and CSA (Clang Static Analyzer) expert. Your task is to analyze a CSA bug report execution path and identify gaps — places where the analyzer made overly-simplified assumptions, missed function bodies, or failed to explore branches — that could make this a false positive.

A "gap" is a point in the execution path where the analyzer's reasoning is incomplete AND that incompleteness could make the path a false positive:
1. **overly_simple_assumption**: The analyzer assumed a condition is true/false without considering the full guard (e.g., "Assuming the condition is true" when the condition has multiple parts).
2. **missed_function_body**: A function call was treated as opaque; its return value or side effects were not modeled. **Only report this when the opaque function's actual behavior (from the summary) would make the path infeasible or prevent the bug.** If the summary shows the path is still feasible, do NOT list this gap.
3. **unconstrained_value**: A variable was treated as fully symbolic when its actual range is constrained by the context.
4. **missed_branch**: A branch that would prevent the bug was not explored.

{data_flow_section}

## CRITICAL: Control Event Feasibility Analysis

Some functions in the "Function Summaries" section are marked with **[CONTROL-CONTEXT]**, meaning they were detected in [control] events (e.g., "Taking true branch", "Assuming the condition is true/false"). These functions come with **return_value_constraints** that describe the relationship between inputs and return values.

For EACH control event in the path that involves a function call's return value:
1. Look up the function in the summaries — check if it has ``return_value_constraints``.
2. Check ``return_value_constraints.failure_nature``:
   - ``"deterministic"``: The failure condition depends on program state (e.g., "queue is full"). You CAN check whether the condition holds at the current point in the path.
   - ``"non_deterministic"``: The failure depends on system resources (e.g., "malloc returns NULL"). This type of failure can happen **at any call** and **cannot be ruled out** by program state alone. The control event assumption is **always feasible** — skip the infeasibility check.
3. For **deterministic** failures only: determine whether the "Assuming condition is true/false" is **feasible**:
   - If ``return_value_constraints.success_conditions`` lists conditions that ARE met by the current state, then a "success" return (e.g., return 0) IS feasible.
   - If ``return_value_constraints.failure_conditions`` lists conditions that are NOT met by the current state (e.g., queue NOT full), then a "failure" return (e.g., non-zero) is **infeasible**.
4. If the CSA's "Assuming" condition is infeasible (deterministic failure only), this strongly indicates the path is a **false positive**.
5. Even if a path is a true positive, the control event feasibility analysis helps identify exactly where and why the path can actually execute.

**Example (deterministic)**: If event says "Taking true branch" at L400 where ``if(wslay_queue_push(...) != 0)`` and the summary says ``failure_conditions=["queue is full"]`` with ``failure_nature="deterministic"``, but the queue was just created and is empty, then "Taking true branch" is infeasible — this is a false positive.

**Example (non-deterministic)**: If the same ``wslay_queue_push`` has ``failure_conditions=["malloc returns NULL"]`` with ``failure_nature="non_deterministic"``, then "Taking true branch" is **always feasible** — malloc can always fail due to OOM, regardless of program state. Do NOT flag this as a gap.

The key question: "Are the [control] events' assumptions about function return values consistent with the functions' actual input-output behavior?"

## IMPORTANT: Interpreting "Assuming" Events as Evidence of Function Modeling

When a path event says **"Assuming field 'X' is <value>"** or **"Assuming the condition is true/false"** at a specific source line, this means CSA **DID enter and model that function's body** — it reached a conditional statement inside the function and made a branch choice. Pay attention to the **source code at the event's line number** (provided in the Source Code Context sections) to see the actual condition.

Use this to distinguish two scenarios:
- **Events INSIDE a function body** (e.g., ``"Assuming field 'blockLength_' is <= 0"`` at a line where ``if (blockLength_ > 0)`` appears): The function was **NOT opaque**. CSA entered it, evaluated its condition, and chose a specific branch. The gap, if any, is about **whether that branch choice is realistic** (``unconstrained_value`` / ``overly_simple_assumption``), NOT that the body was missed.
- **Only "Calling ... →" and "← Returning from ..." events** with nothing in between: The function was **truly opaque**. ``missed_function_body`` may be appropriate here.

Also note the event kind in the path listing:
- ``[event]`` events describe facts (assignments, assumptions, calls). An "Assuming X" event marked ``[event]`` still means CSA evaluated a condition — it means the condition was **used for tracking** but not necessarily a branch divergence (that would be ``[control]``).
- ``[control]`` events describe explicit branch decisions ("Taking true/false branch").

**Example**: If a path has ``#5 [event] L62: ← Assuming field 'blockLength_' is <= 0 →``, and the source at line 62 shows ``if (blockLength_ > 0 && totalLength_ > 0)``, then:
- CSA **entered the constructor body** and evaluated the if-condition.
- It chose the branch where ``blockLength_ <= 0``, skipping the allocation.
- The correct gap type is **``unconstrained_value``** (CSA didn't constrain ``blockLength_`` to realistic values from the caller) or **``overly_simple_assumption``**, NOT ``missed_function_body``.

You should respond ONLY with valid JSON. No markdown fences, no extra text."""

GAP_ANALYSIS_PROMPT = """Analyze the following CSA false-positive (FP) report execution path for gaps.

## Bug Information
- **Type**: {bug_type}
- **Location**: {bug_location}
- **Function**: {bug_function}
- **Description**: {bug_description}

## Source Code Context
```
{source_context}
```

## Execution Path ({num_events} events)
{error_start_legend}
→ = Function call
```
{path_events}
```
{error_start_info}

{call_chain_context}
## Task

{data_flow_phase1}

### Phase 2: Gap Identification
Then, list gaps following the standard format. **Only list a gap if it actually contributes to making the path a false positive.** If the data flow analysis shows the path is still feasible despite the analyzer's incomplete reasoning, do NOT list a gap.

Key guidelines:
- A ``missed_function_body`` gap means the function's actual behavior would prevent the bug or make the path infeasible. If the function summary shows the path is still feasible (e.g., non-deterministic failure like OOM), do NOT list it.
- A ``missed_function_body`` gap with non-deterministic failure: severity should be **low** at most, since the failure can always occur.
- If the only gaps you can find don't actually make the path a false positive, leave the ``gaps`` list empty and set ``is_likely_fp`` to ``false``.

Ensure your gaps reflect the data flow analysis above — if the root cause is an opaque function that would have initialized the value, the corresponding gap must mention that function specifically, not a downstream function.

### Phase 3: Output
Provide:
1. `is_likely_fp`: whether this path is likely a false positive
2. `confidence`: your confidence ("high", "medium", "low")
3. `reasoning`: a short summary including your data flow root cause analysis
4. `gaps`: list of gaps found

For each gap, provide:
1. `gap_type`: one of "overly_simple_assumption", "missed_function_body", "unconstrained_value", "missed_branch"
2. `location`: where the gap occurs (event number or function:line)
3. `description`: what the analyzer missed
4. `severity`: "high", "medium", or "low"
5. `source_context`: the relevant code snippet (if any)
6. `suggestion`: what would need to be true for the path to be correct

## Output Format
```json
{{
  "path_id": "report-name",
  "is_likely_fp": true,
  "confidence": "high",
  "reasoning": "Data flow analysis: ... Overall analysis...",
  "gaps": [
    {{
      "gap_type": "missed_function_body",
      "location": "event #12 at line 145",
      "description": "The analyzer treats to_ascii_decimal as opaque...",
      "severity": "high",
      "source_context": "char buf[32]; to_ascii_decimal(buf, val);",
      "suggestion": "Model to_ascii_decimal as writing to buf..."
    }}
  ]
}}
```"""


# ---------------------------------------------------------------------------
# Prompt 2: Path Completion
# ---------------------------------------------------------------------------

PATH_COMPLETION_SYSTEM_PROMPT = """You are a C++ static analysis and POC generation expert. Your task is to take a CSA bug report path that has been analyzed for gaps, and "complete" the path by filling in the missing information — additional branch conditions, function calls, variable concretizations, and preconditions — that would make the path a valid reproduction of a real bug.

You should respond ONLY with valid JSON. No markdown fences, no extra text."""

PATH_COMPLETION_PROMPT = """Complete the following CSA execution path by filling in the identified gaps.

## Original Path Summary
{original_summary}

## Bug Information
- **Type**: {bug_type}
- **Location**: {bug_location}:{bug_line}
- **Function**: {bug_function}

## Gap Analysis
{gap_analysis}

## Source Code Context
```
{source_context}
```

## Execution Path
```
{path_events}
```

## Task
For each identified gap, provide "completed steps" that fill in the missing information. Each step should include:
1. `original_event_index`: which event in the original path this relates to (or null for a completely new step)
2. `new_event_kind`: "event", "control", or "assume"
3. `description`: what happens in this step
4. `file_path`: the source file
5. `line_number`: the line number (0 for synthetic)
6. `code_snippet`: relevant code
7. `concretization`: any concrete values for variables (e.g., {{"ptr": "nullptr", "size": 128}})

Also provide:
- `original_path_summary`: one-sentence summary of what the original path does
- `final_path_summary`: one-sentence summary of the completed path
- `preconditions`: list of conditions that must hold for the bug to trigger
- `concretized_values`: map of variable names to concrete values

## Output Format
```json
{{
  "original_path_summary": "The analyzer path goes from A to B, assuming X...",
  "final_path_summary": "With the completed path, caller does Y which leads to...",
  "preconditions": ["condition1", "condition2"],
  "concretized_values": {{"var1": "value1"}},
  "completed_steps": [
    {{
      "original_event_index": null,
      "new_event_kind": "event",
      "description": "Caller invokes setFromLocalAddr with a valid addrinfo list",
      "file_path": "SocketAddress.cpp",
      "line_number": 645,
      "code_snippet": "setFromSockaddr(info->ai_addr, ...)",
      "concretization": {{"info": "valid addrinfo pointer"}}
    }}
  ]
}}
```"""


# ---------------------------------------------------------------------------
# Prompt 3: POC Generation
# ---------------------------------------------------------------------------

POC_GENERATION_SYSTEM_PROMPT = """You are a C++ engineer specializing in writing minimal reproducer test cases for bugs found by static analysis. Given a completed execution path for a bug, generate a complete, compilable C++ test case that demonstrates the execution path.

Rules:
- Output ONLY valid C++ code, no explanations.
- The code must be self-contained (include all headers).
- Use gtest or standalone main() as appropriate.
- Focus on minimal code that exercises the exact buggy function and file.
- Match the coding style of the target project ({project_name} in this case).
- Do NOT generate tests for unrelated functions — only the buggy function.
- If the bug is in a third-party header (e.g., gtest), create a test in the project's source file that demonstrates how the calling context triggers the path.

REACHABILITY FROM A REAL PUBLIC ENTRYPOINT (critical — this decides TP vs FP):
The POC must demonstrate that a REAL caller, invoking a REAL public API of {project_name}, reaches the flagged statement through genuine control/data flow and fires the bug. A POC that reaches the buggy statement ONLY via a manufactured shortcut is a POC DEFECT and proves nothing about whether the bug is real.

The "Bug Function" may be an internal helper, a virtual method, or a template — NOT necessarily a public entrypoint. When it is, call the REAL public API/entrypoint that legitimately reaches it, and set up the real caller's state (allocated objects, non-null output buffers), not the pathological state as a literal.

Prohibited shortcuts (each fabricates reachability and must NOT be used):
(a) Do NOT call the buggy internal/leaf/template function directly with a fabricated null / contradictory argument. E.g. do not write `heap_heapify(..., /*bh_ids=*/nullptr, ...)`, `scanner->scan_codes(..., /*ids=*/nullptr, ...)`, or any direct call that hands a null literal to a parameter real callers never pass null to. Drive the real caller (e.g. `Index::search`, `IndexRefine::search`) that allocates and passes a valid buffer, and reach the helper through it.
(b) Do NOT construct the buggy object/container in a degenerate empty state that real code cannot reach, then operate on its null internal pointer (e.g. build a zero-size `AlignedTable` whose `ptr` is null and call `clear()`/`memset` on it). If every real caller allocates size > 0 before that operation, size it > 0 and reach it through the real allocation path.
(c) Do NOT defeat virtual dispatch by defining your own subclass that omits an override so an overridable base method runs. Call sites dispatch on the object's actual dynamic type; if every real subclass overrides the flagged virtual, the base default is unreachable → the report is an FP, not a POC to fabricate.
(d) Do NOT re-implement or stub the real library function under `namespace {project_name}` (or otherwise) with a made-up internal format/magic. Call the REAL function on a REAL input it can actually parse; do not invent a header/magic the real parser rejects (e.g. a deserializer called on a buffer it cannot read).
- A genuine TP POC is one whose triggering null/invalid condition is PRODUCED by real inputs flowing through real callers — not hard-coded as a literal argument or by constructing the exact pathological state directly in the test's own setup. If the only way to reach the flagged statement is a fabricated state/entry that no real caller can produce, the bug is UNREACHABLE in real usage → the report is a False Positive (do not fabricate a POC for it).
{oom_rules}"""

POC_GENERATION_PROMPT = """Generate a minimal C++ test that calls the exact buggy function.

## Bug Information
- **Type**: {bug_type}
- **Location**: Bug file: {bug_location}, line {bug_line}
- **Function**: {bug_function}

## Completed Path Summary
{completed_path_summary}

## Preconditions to Set Up
{preconditions}

## Concretized Values (use these as actual C++ values)
{concretized_values}

## Source Code of the Buggy Function
```cpp
{source_context}
```

## Target Project
{project_name} project.

{project_specific_context}

## CRITICAL Requirements
1. Generate a single C++ file with a gtest TEST() case
2. Include all necessary headers (project-specific and standard)
3. Reach the flagged statement "{bug_function}" (at {bug_location}:{bug_line}) by calling a REAL public entrypoint of {project_name} that drives execution into it through genuine control/data flow. If {bug_function} is an internal helper / virtual method / template (not a public API), call the real public method that reaches it and set up a real caller's valid state — do NOT call the internal function directly with a fabricated null/hostile argument
4. Set up preconditions using the project's APIs (allocate real objects/buffers as a real caller would; do NOT hard-code the null/invalid condition as a literal, and do NOT build the buggy object in an empty/zero-size state that no real caller reaches)
5. Show the code path that the static analyzer flagged
6. Only output the code, no explanations
7. If the bug is in a template/header, include that header and trigger the instantiation through the real public API that instantiates it
8. Do NOT re-implement/stub the real library function with a made-up internal format, and do NOT subclass to drop a virtual override so a base-class default runs — real dispatch must be preserved
9. If the flagged statement can ONLY be reached through a fabricated state/entry that no real caller can produce (e.g. passing null to a parameter every real caller passes non-null, or an empty object no caller constructs empty), the report is a False Positive: do NOT fabricate — explain that instead of emitting a fake POC

{oom_trigger_instruction}

```cpp
#include <gtest/gtest.h>
// ... other includes ...

TEST(/* suite */, /* name */) {{
  // setup
  // call the buggy function
}}
```"""


OOM_RULES = """- If the bug trigger involves heap memory allocation failure (e.g., malloc returns NULL inside an opaque function like wslay_queue_push), you MUST include a malloc interposer in the POC to trigger the bug deterministically.
- The interposer uses `dlsym(RTLD_NEXT, "malloc")` with a call counter and recursion guard (see OOM Trigger Mechanism section for the exact pattern).
- Make the failure threshold configurable via a global variable, and loop through possible thresholds to find the triggering one.
- After the trigger section, ALWAYS restore `g_malloc_fail_after = 0` so subsequent allocations (including gtest infrastructure) work normally.
- For API initialization functions (e.g., wslay_event_context_server_init), provide a valid (non-null) callbacks struct with no-op stubs — never pass nullptr."""

OOM_TRIGGER_INSTRUCTION = """## OOM / Malloc Failure Trigger Mechanism
If the bug trigger requires malloc failure (e.g., a memory leak when an internal malloc fails mid-stream), you MUST include a malloc interposer to trigger the bug deterministically.

The interposer uses ``dlsym(RTLD_NEXT, "malloc")`` with a call counter. A recursion guard prevents infinite loops during the initial ``dlsym`` call (which itself calls ``malloc``):

.. code-block:: cpp

    #define _GNU_SOURCE
    #include <dlfcn.h>
    #include <cstddef>

    static size_t g_malloc_fail_after = 0;   // 0 = never fail
    static size_t g_malloc_counter = 0;

    extern "C" void *malloc(size_t size) noexcept {
        static void *(*real_malloc)(size_t) = nullptr;
        static int resolving = 0;

        if (g_malloc_fail_after > 0 && ++g_malloc_counter >= g_malloc_fail_after) {
            return nullptr;
        }
        if (!real_malloc) {
            if (resolving) {
                extern "C" void *__libc_malloc(size_t);
                return __libc_malloc(size);
            }
            resolving = 1;
            real_malloc = (void *(*)(size_t))dlsym(RTLD_NEXT, "malloc");
            resolving = 0;
        }
        return real_malloc(size);
    }

CRITICAL requirements:
1. Define the interposer BEFORE any ``#include`` that transitively includes ``<cstdlib>`` / ``<stdlib.h>``, to avoid declaration conflicts with the system's ``malloc``.
2. Use the recursion guard shown above — without it, ``dlsym(RTLD_NEXT, "malloc")`` will recurse infinitely and crash with a stack overflow.
3. The interposer declaration must use ``noexcept`` to match glibc's ``__THROW`` attribute on ``malloc`` (otherwise ``-fsyntax-only`` will report an exception-specification mismatch).
4. After the code section that needs controlled malloc failure, ALWAYS restore normal allocation:
   ``g_malloc_fail_after = 0; g_malloc_counter = 0;``
   Otherwise gtest's own infrastructure (``EXPECT_*``, test teardown) will crash trying to allocate.
5. Loop through possible failure thresholds (e.g., ``for (size_t fp = 1; fp <= 50; ++fp)``) to find the one that triggers the bug.
   IMPORTANT: Do NOT break out of the loop on error (do NOT write ``if (r != 0) break;``). The bug only triggers at a specific fp value; breaking early will skip it. Continue the loop for all fp values so ASAN can detect leaked memory at process exit.
6. For API initialization functions that take a callbacks parameter (e.g., ``wslay_event_context_server_init``), provide a valid non-null callbacks struct with no-op stubs. These functions will crash with a segfault if passed ``nullptr`` for callbacks — they dereference the pointer unconditionally.
"""


# ---------------------------------------------------------------------------
# Prompt 4: Gap Resolution Verification (completion loop)
# ---------------------------------------------------------------------------

VERIFICATION_SYSTEM_PROMPT = """You are a C++ static analysis expert. You will receive:
1. The original CSA execution path for a bug report.
2. The gap analysis identifying what the analyzer missed.
3. "Completed steps" that fill in those gaps with concrete values, branch conditions, and opaque function expansions.

Your task is to determine whether the completed path is a TRUE POSITIVE (the bug is real and reachable) or a FALSE POSITIVE (after filling gaps, the path is clearly unreachable or the bug cannot manifest).

A TRUE POSITIVE means:
- All identified gaps have been addressed.
- The path, with gaps filled, leads to a real exploitable bug.
- The preconditions are satisfiable in real code.

A FALSE POSITIVE means:
- Even after filling gaps, the path still cannot lead to the bug in real execution.
- The gap filling shows that the analyzer's assumptions were wrong (e.g., the opaque function would have initialized the value).
- The preconditions are contradictory or unreachable.

You should respond ONLY with valid JSON. No markdown fences, no extra text."""

VERIFICATION_PROMPT = """Verify whether the completed CSA path is a true positive or false positive.

## Bug Information
- **Type**: {bug_type}
- **Location**: {bug_location}
- **Description**: {bug_description}

## Original Path Events
```
{path_events}
```

## Gap Analysis
{gap_analysis}

## Completed Steps (filling the gaps)
{completed_steps}

## Task
1. Are ALL identified gaps now addressed by the completed steps? Check each gap individually.
2. With all gaps addressed, does the execution path still lead to the reported bug? Consider:
   - Are the preconditions actually satisfiable?
   - Do the concretized values make sense?
   - Does the opaque function's unmodeled behavior (now filled in) prevent the bug?
3. Provide your final classification.

## Output Format
```json
{{
  "all_gaps_addressed": true,
  "classification": "false_positive",
  "explanation": "After completing the path, the root cause is that ... The opaque function would have initialized the value, so the bug cannot occur in real execution.",
  "remaining_issues": "If not all gaps addressed, what is still missing"
}}
```"""


# ---------------------------------------------------------------------------
# Prompt 5: Compile Fix Iteration
# ---------------------------------------------------------------------------

COMPILE_FIX_SYSTEM_PROMPT = """You are a C++ engineer fixing compilation errors in a POC test case. You will receive:
1. The original POC source code that failed to compile.
2. The compilation error messages from clang++-14.

Your task is to output a corrected version of the POC that compiles successfully.

Important project context:
{project_hints}

Output ONLY valid C++ code. No explanations, no markdown fences."""

COMPILE_FIX_PROMPT = """Fix the following C++ POC code so it compiles successfully.

## Original POC Code
```cpp
{source_code}
```

## Compilation Errors
{compile_errors}

{type_context}
{fix_history}
## Task
Fix all compilation errors listed above. Output the corrected C++ code only.
- Keep the same test structure and intent.
- Look at EACH error and fix its root cause (wrong API, missing include, wrong namespace, wrong argument types, etc.).
- Fix includes, types, namespaces, and API usage as needed.
- Do NOT change the fundamental test logic — the test should still exercise the target function.
- Make sure the headers included actually exist in the target project or the system.
- Your output must be DIFFERENT from the original code — you must make changes to fix the errors.
- If you are unsure about an API, make your best guess based on the project context provided.
- Output ONLY valid C++ code, no explanations."""


# ---------------------------------------------------------------------------
# Prompt 6: Function Summary Generation
# ---------------------------------------------------------------------------

FUNCTION_SUMMARY_SYSTEM_PROMPT = """You are a C++ static analysis expert. Your task is to analyze a C/C++ function's body and produce a structured, bug-type-specific summary that answers the question: "If the Clang Static Analyzer (CSA) could not model this function's body, what critical information would the analyzer miss?"

The summaries are used to fill in gaps when CSA treats a function as opaque. The output must be precise, factual, and focused on the specific bug type.

You should respond ONLY with valid JSON. No markdown fences, no extra text."""

FUNCTION_SUMMARY_PROMPT = """Analyze the following C/C++ function and produce a structured summary for the "{bug_type}" bug type.

## Function to Analyze
- **Name**: {function_name}
- **File**: {file_path}
- **Signature**: `{signature}`

## Function Body
```c
{function_body}
```

## Task
Analyze this function and produce a JSON summary with:
1. `analysis` (str): Concise summary of the function's behavior relevant to this bug type. Focus on what CSA would miss by not modeling this function.
2. `preconditions` (list[str]): Conditions that must hold for correctness (e.g., "ptr != NULL", "len <= bufsize").
3. `return_value_constraints` (dict or null): If the function's return value affects control flow, describe:
   - `return_value_semantics`: What different return values mean (e.g., "0 = success, -1 = error")
   - `success_conditions`: When does the function succeed?
   - `failure_conditions`: When does the function fail?
   - `failure_nature`: "deterministic" (state-based) or "non_deterministic" (resource/OS failure)

## Output Format
Output ONLY valid JSON with these three fields. No markdown fences, no extra text."""

FUNCTION_SUMMARY_BEST_GUESS_PROMPT = """You are analyzing a function from the "{project_name}" project for which the actual source code body is not available. Based on the function name, signature, and any knowledge you have of this library/project, provide a best-guess structured summary for the "{bug_type}" bug type.

## Function to Analyze
- **Name**: {function_name}
- **File**: {file_path}
- **Signature**: `{signature}`

## Task
1. Based on the function name, signature, and your knowledge of common C/C++ libraries, infer what this function likely does.
2. Produce a JSON summary with:
   - `analysis` (str): Best-guess analysis of behavior. State clearly that this is a best guess.
   - `preconditions` (list[str]): Conditions that must hold for correctness.
   - `return_value_constraints` (dict or null): If the return value affects control flow, describe its semantics.
3. **CRITICAL**: Set `is_best_guess` to `true` since this is an inference.
4. Be conservative — if uncertain, note the uncertainty in the analysis.

## Output Format
Output ONLY valid JSON. No markdown fences, no extra text."""


# ---------------------------------------------------------------------------
# Prompt 7: CFG-based Branch Feasibility Analysis (In-Report FP)
# ---------------------------------------------------------------------------

CFG_FEASIBILITY_SYSTEM_PROMPT = """You are a C++ control-flow analysis expert. Given a segment of a CSA execution path:

1. **Precondition** — three-part context:
   (a) **Accumulated state** from all prior segments
   (b) **Previous segment's postcondition** (the branch outcome)
   (c) **Function summaries** for each function called in this segment
2. **Program Path**: the C++ source code in this segment
3. **Postcondition**: the branch outcome that CSA chose

Your task: determine whether the program path CAN produce the postcondition GIVEN the precondition.

### Key Rules
- Use function summaries as FACTS (not guesses) when evaluating if-conditions that involve function calls.
- A function summary's ``return_value_constraints`` tells you the success/failure conditions.
- If reaching the postcondition requires contradictory variable values (e.g. ``x > 0`` AND ``x <= 0``), the path is INFEASIBLE.
- If ALL if-conditions independently match the postcondition, the path is FEASIBLE.

Output ONLY valid JSON."""

CFG_FEASIBILITY_PROMPT = """## Segment Feasibility Analysis

### Bug Context
- **Bug Type**: {bug_type}
- **File**: {bug_file}:{bug_line}
- **Function**: {bug_function}

### Precondition

#### (a) Accumulated State (from previous segments)
```
{accumulated_state}
```

#### (b) Previous Segment's Postcondition
```
{prev_postcondition}
```

#### (c) Function Summaries (supplementary — from cache, no automatic generation)
{function_summaries}

#### (d) Project-wide Constraints (invariants mined from source)
{global_constraints}

### Program Path (C++ source code in this segment)
```cpp
{source_context}
```

### Postcondition (CSA's chosen branch outcome)
```
{postcondition}
```

### Task
Given the precondition (accumulated state + previous postcondition + function summaries) as established facts, can the **program path** produce the **postcondition**?

Trace through the code line by line:
1. Start with all precondition facts as true.
2. At each if-statement, determine which branch the current facts dictate.
3. Record any new facts established by assignments or function returns.
4. Check if you can reach the postcondition without contradictions.

**Output format:**
```json
{{
  "feasible": true,
  "established_state": "After this segment: r >= 0, ctx->read_enabled is true, ctx->server == 0 (key variable values and constraints established by this segment's code execution)",
  "reasoning": "Step-by-step trace through the code showing how facts lead to postcondition...",
  "contradiction": null
}}
```

If infeasible:
```json
{{
  "feasible": false,
  "established_state": null,
  "reasoning": "Step-by-step trace showing where the contradiction occurs...",
  "contradiction": "Variable X must satisfy both A and B at line N"
}}
```

The ``established_state`` field is **critical**: it describes the STATE AFTER this segment executes.  It must include:
- All variable values that were determined by the code in this segment
- The cumulative effect of ALL previous branches taken so far
- Any constraints implied by the precondition AND the code logic

This ``established_state`` will be passed as the ``precondition`` for the **next** segment.
"""


# ---------------------------------------------------------------------------
# Prompt 8: Constraint Completion — identify + fill ALL implicit constraints
# ---------------------------------------------------------------------------

CONSTRAINT_COMPLETION_SYSTEM_PROMPT = """You are a C++ static analysis expert. Your task is to take a CSA (Clang Static Analyzer) bug report execution path and identify EVERY implicit assumption, unconstrained variable, and opaque function call that could affect the feasibility of the path.

For each implicit assumption, provide concrete values or constraints that would make the path fully determined. Do NOT assume any constraint is "obvious" — spell everything out.

The output must make every branch condition, variable value, function return value, and memory state explicit so that the path can be evaluated for logical consistency.

CRITICAL — Disambiguate variables on different objects: C++ code often has member fields with the SAME name on DIFFERENT objects (e.g. `iocb.opcode` vs `ctx->imsg->opcode`). When the CSA path says "Assuming field 'opcode' is X", you MUST check the source code context to determine WHICH object's opcode is being referenced. Do NOT assume that a constraint on `iocb.opcode` also applies to `imsg->opcode`, or vice versa — they are completely independent variables.

You should respond ONLY with valid JSON. No markdown fences, no extra text.

CONCISENESS (critical — the response must FIT within the token limit): keep every string field short and factual. Do not restate path events verbatim. In `completed_steps`, include only the steps that resolve an actual implicit assumption (omit boilerplate); keep `code_snippet` to a single line and omit it entirely when the description is self-explanatory. A shorter response is always preferred over a complete-but-truncated one."""

CONSTRAINT_COMPLETION_PROMPT = """Complete the following CSA execution path by identifying and filling ALL implicit constraints.

Unlike traditional gap analysis (which only fills pre-identified gaps), you must enumerate EVERY implicit assumption in the path and provide concrete values or constraints that would make the path fully specified.

## Bug Information
- **Type**: {bug_type}
- **Location**: {bug_location}:{bug_line}
- **Function**: {bug_function}
- **Description**: {bug_description}

## Source Code Context
```
{source_context}
```

## Execution Path ({num_events} events)
{error_start_legend}
→ = Function call
```
{path_events}
```
{error_start_info}

## Task

### Phase 1: Constraint Identification
For EACH event in the path, identify what is implicit or underspecified:

1. **Branch control events** ("Taking true/false branch", "Assuming condition"): What are the actual variable values that produce this branch outcome? Are the variables constrained by prior calls or preconditions?
2. **Opaque function calls** ("Calling ...", "← Returning from ..."): What does the function do? What are its preconditions? What does it return under what conditions?
3. **Event descriptions mentioning symbolic values** (e.g., a "garbage value", "uninitialized value", or symbolic pointer): What concrete constraints should apply to this value in real execution?
4. **Implicit preconditions**: What must be true about the program state before this path is taken?

### Phase 2: Constraint Concretization
For each identified implicit assumption:
- Assign a concrete value or range where possible
- If a value cannot be fully determined, specify its constraints (e.g., "blockLength_ > 10 && blockLength_ <= 256")
- For opaque function return values, specify based on the actual function behavior

### Phase 2.5: CRITICAL — Check if Error Start Variable Is Initialized via Function Parameter

The Error Start event marks a variable that the CSA says is uninitialized. This is often a FALSE POSITIVE when:

**Pattern A: Variable declared in a loop, passed by pointer to a function that initializes it**
Example:
```c
for(...) {{
    uint32_t var;           // ← CSA says var is uninitialized
    func(&var, ...);        // func writes to *var on some code path
}}
```
If `func()` writes to `*var` in ANY code path (e.g., when its internal state machine is in an initial state), then `var` IS initialized by the first call. On subsequent iterations, `var` retains the value from the previous call. The CSA treats each iteration independently and misses this cross-iteration initialization.

**Pattern B: State reset function initializes state between messages/calls**
If the function that writes to `*var` has an internal state variable (like `utf8state`) that is:
- Set to a known initial value (like `UTF8_ACCEPT`) by a state reset function
- Called reliably before each new message/operation

Then on the FIRST call with fresh state, `*var` IS set (not read before write).

**To determine if the Error Start is a false positive:**
1. Find the variable declaration in the source code context
2. Check if the variable's address is passed to any called function
3. Look at the called function's body (if included in source context): does it write to `*param` via one branch?
4. Does the called function have a state parameter that is initialized to a known value by external code?
5. **Compare reset vs non-reset functions**: Some functions in the path may SET the state variable (e.g., `imsg_reset` sets `utf8state = UTF8_ACCEPT`), while others DO NOT touch it at all (e.g., `imsg_set` does not modify `utf8state`). If a reset function is called before the Error Start, but a non-reset function is called in between, the state variable RETAINS its initial value — the non-reset function does not corrupt it.
6. If the variable is initialized through any pointer write path → `error_start_is_real: false`

### Phase 3: Output
Provide the completed path with:

1. `original_path_summary`: one-sentence summary of the original path
2. `final_path_summary`: one-sentence summary with all constraints applied
3. `preconditions`: full list of conditions that must hold
4. `concretized_values`: map of variable names to concrete values or constraints
5. `error_start_is_real`: boolean — is the Error Start value truly uninitialized? (false = initialized via memcpy/memset/loop-init that CSA doesn't model → false positive)
6. `completed_steps`: additional steps that make the implicit explicit

## Output Format
```json
{{
  "original_path_summary": "Original path: ...",
  "final_path_summary": "After constraint completion: ...",
  "preconditions": [
    "blockLength_ > 0",
    "totalLength_ > 0"
  ],
  "concretized_values": {{
    "blockLength_": "> 0",
    "totalLength_": "> 0"
  }},
  "error_start_is_real": true,
  "completed_steps": [
    {{
      "original_event_index": null,
      "new_event_kind": "event",
      "description": "Constructor allocates bitfield_ because blockLength_ > 0 && totalLength_ > 0",
      "file_path": "DefaultPieceStorage.cc",
      "line_number": 42,
      "code_snippet": "bitfield_ = new unsigned char[blockLength_];",
      "concretization": {{"bitfield_": "non-null allocated pointer"}}
    }}
  ]
}}
```

## CRITICAL: Cross-Check Concretizations Against Branch Decisions

After filling concretized_values, VERIFY that every concretized value is CONSISTENT with the path's branch decisions:

- If the path takes a **FALSE branch** at `if(wslay_is_ctrl_frame(iocb.opcode))`, then `iocb.opcode` MUST NOT be a control frame.
- If the path takes a **TRUE branch** at `if(ctx->imsg->opcode == WSLAY_PING)`, then `ctx->imsg->opcode` MUST be WSLAY_PING.
- These are DIFFERENT objects (`iocb.opcode` ≠ `imsg->opcode`), so they can have different values simultaneously.

For EACH concretized value in the output, trace which branch conditions constrain it. If a concretization contradicts ANY branch decision, it's WRONG — fix it before outputting.```"""


# ---------------------------------------------------------------------------
# Prompt 9: Feasibility Check — verify completed constraints are consistent
# ---------------------------------------------------------------------------

FEASIBILITY_CHECK_SYSTEM_PROMPT = """You are a C++ formal verification expert. Given a CSA execution path, its completed constraints, and the source code context, your task is to determine whether the CSA bug warning is a TRUE POSITIVE (real bug) or FALSE POSITIVE (analyzer limitation or infeasible path).

A warning is a **FALSE POSITIVE** if:
- **Constraint contradiction**: Two or more constraints require the same variable to have contradictory values (e.g., `x > 0` AND `x <= 0`)
- **Precondition conflict**: Preconditions are mutually exclusive (e.g., "blockLength_ > 0" AND "blockLength_ <= 0")
- **Function return contradiction**: The path requires a function to simultaneously return success and failure
- **Value actually initialized**: The variable or memory that CSA flagged as uninitialized/garbage IS actually initialized before use through a mechanism CSA may not model (e.g., memcpy, memset, or a prior assignment in the path that CSA didn't track)
- **Degenerate/unreachable scenario**: The concretized values would require a logically impossible program state (e.g., an empty handler table AND a successful long-option parse via getopt_long)
- **Caller constraints violate the path**: Real-world invariants from the call site make the path impossible (e.g., a constructor parameter is always > 0 in practice)

A warning is a **TRUE POSITIVE** if:
- All constraints are consistent AND
- The flagged variable/value IS genuinely uninitialized/invalid at the point of use AND
- No memcpy, memset, or other initialization mechanism writes to the flagged variable before use

You should respond ONLY with valid JSON. No markdown fences, no extra text."""

FEASIBILITY_CHECK_PROMPT = """Determine whether the CSA bug warning at {bug_location}:{bug_line} is a TRUE POSITIVE (real bug) or FALSE POSITIVE (analyzer limitation). This is a two-part check: (A) constraint consistency, and (B) validity of the specific CSA warning.

## Bug Information
- **Type**: {bug_type}
- **Location**: {bug_location}:{bug_line}
- **Function**: {bug_function}
- **Description**: {bug_description}

## Error Start (Value/Event That CSA Flags)
The variable or event that CSA considers problematic originates here:
{error_start_info}

## Source Code Context (at the bug site and surrounding code)
```
{source_context}
```

## Original Path Summary
{original_path_summary}

## Completed Constraints (filled by constraint completion)
{completed_constraints}

## Preconditions for the Path
{preconditions}

## Concretized Values
{concretized_values}

{previous_conflicts}

## Task — Two-Part Analysis

### Part A: Constraint Consistency (existing)
1. **Check each precondition**: Can ALL preconditions be simultaneously true?
2. **Check concretized values**: Do any concretized values contradict each other or the preconditions?
3. **Trace the logic**: Walk through the path events and the completed steps. At each branch point, ask: "Given the current state, can this branch be taken?"
4. **Check for logical consistency**: Are there any variables that must satisfy both X and NOT X?

### Part B: CSA Warning Validity (CRITICAL — do not skip)
1. **Identify the flagged variable/value**: What variable or memory location does CSA claim is uninitialized/garbage/null at {bug_location}:{bug_line}?
2. **Trace initialization**: Walk backward from the USE point. Is the variable initialized BEFORE use? Consider:
   - Direct assignment
   - **memcpy/memset** that writes to the variable (common CSA blind spot)
   - Constructor initialization
   - Function calls that write through a pointer
3. **Check if the warning is a known CSA FP pattern**:
   - CSA does not model memcpy/memset as initializers — if memcpy wrote to the variable before use, CSA's warning is a FALSE POSITIVE
   - CSA does not model certain function contracts — if a called function guarantees initialization, CSA may miss it
   - CSA may flag **"dead reads"** — where an uninitialized value IS read, but the read does NOT affect observable program behavior
4. **Critical: Check whether the uninitialized read PROPAGATES to observable behavior**:
   - Is the flagged variable an OUTPUT parameter that the caller uses after the call? Or is it a purely local variable that dies at end of scope?
   - Is the garbage value only used to compute the SAME variable which is then overwritten (e.g., `*ptr = ... (*ptr << ...)` pattern)? In this case the garbage value is consumed and doesn't escape.
   - Does the CALLER of the flagged function actually USE the variable afterwards? Or does it only check the function's return value?
   - Can the garbage value reach: function return value, caller-visible output parameters, global variables, heap memory, or control flow decisions?
   - If the garbage value is read but cannot affect any observable state → **FALSE POSITIVE (dead read)**
5. **Evaluate realism**: Even if constraints are technically consistent, could this scenario actually occur in real execution? Consider:
   - Would getopt_long return 0 when no long options with flag pointers are configured?
   - Would a BitfieldMan be constructed with blockLength_ = 0 in production?
   - Would getifaddrs() return entries with uninitialized sockaddr data?
   - Would a WebSocket close frame's UTF-8 validation enter a multi-byte sequence where codep's prior value affects observable behavior?

### Final Verdict
Combine Part A + Part B: A warning is FALSE_POSITIVE if EITHER:
- Constraints are contradictory (infeasible path), OR
- The flagged variable IS initialized before use (memcpy/memset), OR
- The garbage value is read but does NOT propagate to observable behavior (dead read)

Only TRUE_POSITIVE if BOTH constraints are consistent AND the variable is genuinely uninitialized AND the garbage value propagates to observable state.

## Output Format
Always include `"is_real_bug"` to distinguish "feasible but still FP" from "actually a real bug".
```json
{{
  "is_feasible": true,
  "is_real_bug": false,
  "explanation": "Constraints are consistent, but the variable flagged by CSA at line 1684 IS initialized by memcpy on line 1683 before use. CSA does not model memcpy as an initializer. This is a known CSA FP pattern.",
  "conflicts": []
}}
```

If infeasible (constraint contradiction):
```json
{{
  "is_feasible": false,
  "is_real_bug": false,
  "explanation": "Contradiction: getopt_long cannot return 0 when no flag pointers exist in longOpts.",
  "conflicts": [
    "Precondition: longOpts has only terminator entries (all flag=nullptr)",
    "Event: getopt_long returns 0 (c == 0)",
    "getopt_long returns 0 only when a long option with non-null flag is matched"
  ]
}}
```"""


# ---------------------------------------------------------------------------
# Prompt 10: Single-Path Feasibility Check (reported path only)
# ---------------------------------------------------------------------------
# Refactored version of the feasibility check.  Instead of enumerating CFG
# alternatives, this prompt only asks the LLM to check whether the *reported*
# path's branch conditions (from the CSA bug report) conflict with the
# completed constraints.
# ---------------------------------------------------------------------------

SINGLE_PATH_FEASIBILITY_SYSTEM_PROMPT = """You are a C++ formal verification expert. Given a CSA execution path, its completed constraints, and sliced source code context, determine whether the **reported path** has any constraint contradictions.

This is a focused check — do NOT enumerate alternative execution paths. Only check whether the specific branch decisions in the bug report can be satisfied given the completed constraints.

KNOWN CSA LIMITATIONS that create infeasible paths:

1. **Loop-carried variable initialization**: When a variable is declared INSIDE a loop body and its address is passed to a function (e.g. ``for(...) { Type var; func(&var, ...); }``), the CSA treats each iteration's variable independently. If the FIRST function call initializes the variable (writes through the pointer), subsequent iterations read an initialized value — but the CSA may still report it as "uninitialized" because it creates a fresh variable instance per iteration. This is a FALSE POSITIVE.

2. **Union type-punning**: Writing to one member of a union (via ``memcpy``, assignment, etc.) initializes the storage shared by ALL members. The CSA's type tracking treats union members as independent variables, missing this shared-storage initialization. Common in POSIX networking code (e.g. ``sockaddr_union`` with ``ad.storage`` and ``ad.in``/``ad.in6``).

3. **Path merging**: The CSA path may combine conditions from logically separate/mutually exclusive execution paths. For example, a path where a pointer is NULL (from one branch) but its associated length is non-zero (from a different branch) — this NULL + non-zero combination is infeasible. CRITICAL: If p == nullptr, any associated n (length/size) MUST be 0 or the path is infeasible.

4. **Variable disambiguation**: C++ code commonly has member fields with the SAME name on DIFFERENT objects (e.g. `iocb.opcode` vs `ctx->imsg->opcode`). A completed constraint about `iocb.opcode` does NOT constrain `imsg->opcode`, and vice versa. When checking a branch decision at a specific line, read the actual source code to determine WHICH object's field is being checked. Do NOT confuse `object1.field` with `object2.field` just because they share the `field` name.

CONFLICT VALIDATION RULE — Check variable qualifications before reporting:
When a concretized value appears to contradict a branch decision, FIRST verify that the concretized variable NAME matches the branch decision's actual variable name (check the source code annotation `// source: ...`).

For example:
- Branch at L694: `if(!wslay_event_config_get_no_buffering(ctx) || wslay_is_ctrl_frame(iocb.opcode))` → this checks `iocb.opcode`
- Concretization: `iocb.opcode == WSLAY_PING (0x09)` → same variable `iocb.opcode` → **real conflict** if FALSE branch taken
- Concretization: `ctx->imsg->opcode == WSLAY_PING` → DIFFERENT variable (`imsg.opcode` ≠ `iocb.opcode`) → **NOT a conflict**

If the concretization uses a DIFFERENT object prefix (`iocb` vs `imsg`) than the branch decision's source code, the concretization has the WRONG variable name. In this case, ignore the apparent contradiction — it's not a real conflict. The concretization should be corrected to use the right variable, but the PATH IS STILL FEASIBLE.

A conflict exists when:
- Two or more constraints require the same variable to have contradictory values (e.g., `x > 0` AND `x <= 0`)
- Preconditions are mutually exclusive
- The reported branch decisions (TRUE/FALSE at specific lines) contradict the completed constraints
- Real-world invariants (from project-wide constraint mining) make the path impossible
- The path exhibits any of the known CSA limitations above (loop-carried init, union type-punning, path merging)

Output ONLY valid JSON. No markdown fences, no extra text."""

SINGLE_PATH_FEASIBILITY_PROMPT = """Check whether the **reported CSA execution path** has any constraint contradictions when combined with the completed constraints below.

## Bug Information
- **Type**: {bug_type}
- **Location**: {bug_location}:{bug_line}
- **Function**: {bug_function}
- **Description**: {bug_description}

## Error Start (Value That CSA Flags)
{error_start_info}

## Sliced Source Code Context (statements relevant to this bug path)
```
{source_context}
```

## Reported Path Branch Decisions
These are the specific branch decisions CSA made in the reported execution path:
```
{branch_decisions}
```

## Completed Constraints (from constraint completion)
{completed_constraints}

## Preconditions for the Path
{preconditions}

## Concretized Values
{concretized_values}

{previous_conflicts}

## Task

Check ONLY the reported path (the branch decisions listed above) against the completed constraints:

1. **Branch vs. constraint consistency**: For EACH branch decision in the reported path, check if the completed constraints support the decision. For example, if the path says "Taking TRUE branch at L42 (x > 5)", check whether x CAN be > 5 given the completed constraints.

2. **Precondition compatibility**: Can ALL preconditions be simultaneously true?

3. **Concretization consistency**: Do any concretized values contradict each other or the preconditions?

4. **Real-world invariant check**: Do the completed constraints respect the project-wide invariants?

If ALL branch decisions are consistent with ALL constraints → `"is_feasible": true, "conflicts": []`.
If ANY conflict exists → `"is_feasible": false` with explanation and details.

Output ONLY valid JSON:
```json
{{
  "is_feasible": true,
  "explanation": "All branch decisions are consistent with completed constraints.",
  "conflicts": []
}}
```

Or:
```json
{{
  "is_feasible": false,
  "explanation": "Branch decision at L42 (TRUE, x > 5) conflicts with constraint x <= 0.",
  "conflicts": [
    "L42: taking TRUE branch requires x > 5, but completed constraint says x <= 0 (from constructor parameter bound)"
  ]
}}
```"""


# ---------------------------------------------------------------------------
# Prompt 10.5: Early Feasibility Check (Step 3.5) — lightweight anchor check
# ---------------------------------------------------------------------------
# Called after constraint completion, BEFORE segment-by-segment selection.
# Checks whether reported-path anchor events conflict with completed
# constraints, using deterministic comparison first, then LLM fallback.
# ---------------------------------------------------------------------------

EARLY_FEASIBILITY_SYSTEM_PROMPT = """You are a C++ static analysis expert. You are given a single anchor event from a CSA bug report — a specific branch decision — and the completed constraints for that branch's variable. Determine whether the anchor's branch direction is consistent with the constraints.

This is a FOCUSED check on one specific variable. Do NOT evaluate the entire path, only the single branch decision vs constraint pair shown below.

If the constraint is ambiguous or you cannot determine consistency, return "insufficient_info" rather than guessing.

Output ONLY valid JSON. No markdown fences, no extra text."""

EARLY_FEASIBILITY_PROMPT = """## Early Feasibility Check — Single Anchor Event

### Bug Context
- **Type**: {bug_type}
- **Location**: {bug_location}:{bug_line}

### Anchor Event
- **Line**: {anchor_line}
- **Branch**: {anchor_branch} — `{anchor_condition}`
- **Variable**: {anchor_variable}

### Completed Constraint for This Variable
- **Variable**: {anchor_variable}
- **Constraint**: {completed_constraint}

### Deterministic Pre-check Result
The deterministic pre-check returned: **{deterministic_result}**

### Task
Determine whether the constraint `{completed_constraint}` on `{anchor_variable}` is consistent with the branch decision at line {anchor_line}.

Consider:
1. Does the constraint clearly support or contradict the branch direction?
2. Is the constraint ambiguous (e.g., "allocated" without null/non-null qualification)?
3. Does the constraint refer to the same variable as the branch condition?

### Output Format
```json
{{
  "is_consistent": true,
  "determination": "consistent | contradictory | insufficient_info",
  "reasoning": "The constraint '{completed_constraint}' on '{anchor_variable}' ...",
  "detected_contradiction": null
}}
```

If contradictory:
```json
{{
  "is_consistent": false,
  "determination": "contradictory",
  "reasoning": "Constraint '{completed_constraint}' requires ... but branch condition requires ...",
  "detected_contradiction": "{anchor_variable} must satisfy both X and Y"
}}
```"""


# ---------------------------------------------------------------------------
# Semantic feasibility — range/finite-domain variable conflict (Step 3.5b)
# ---------------------------------------------------------------------------
# The CSA path is checked as a whole (not anchor-by-anchor) for a self-conflict
# on a *guard-pinned / finite-domain* variable.  The report's own source is
# scanned (domain_facts.domain_facts_from_lines) to recover the small constant
# set a variable like the fourcc tag ``h`` is pinned to, which CSA treats as
# free because the constant factory (fourcc) is opaque/out-of-line.  The LLM
# then decides whether the reported path forces such a variable to be
# simultaneously inside and outside its pinned set (e.g. admitted into a deref
# region only when ``h ∈ S``, yet every sub-guard that would assign the
# deref'd pointer is taken FALSE so the pointer stays null).  That is a
# constraint conflict => the reported bug state is unreachable => FP.
#
# NOTE: {bug_type} etc. are .format()-substituted scalars that never contain
# braces.  The C++ skeleton and the domain-fact text are appended RAW after
# formatting in path_analyzer (they may contain '{' / '}').
SEMANTIC_FEASIBILITY_SYSTEM_PROMPT = """You are a C++ static-analysis expert checking a single CSA bug report for a *range / finite-domain variable constraint conflict*.

CSA often models a variable read from external input (a file/stream/parsed header) as able to take ANY value. But the source actually pins it to a small fixed set via equality guards against constants — often produced by an opaque, out-of-line constant factory such as `fourcc("rrot")` (a little-endian uint32 type tag). CSA cannot see this pinning (it treats `fourcc(...)` as an uninterpreted symbol), so it reports as "feasible" a path that is semantically impossible.

Your job: given the reported path's control decisions and the recovered finite domain of the guard variable(s), decide whether the reported bug state forces such a variable to be simultaneously INSIDE and OUTSIDE its pinned set. If so, the bug path is INFEASIBLE — the report is a FALSE POSITIVE.

Rules (be CONSERVATIVE — only conclude "contradictory" when the source genuinely forces the conflict):
1. Identify the guard/finite-domain variable (e.g. `h`) and the exact constant set each of its guards pins it to.
2. A region-entry guard taken TRUE that gates the reported deref (e.g. `if (h == A || h == B || h == C)`) admits the variable into set S = {A, B, C}.
3. If, to reach the deref with a pointer still NULL (its null-initializer never overwritten), the reported path takes FALSE on EVERY sub-guard that would assign that pointer — and those sub-guards' tags partition/exhaust exactly S — then the variable would need to equal one of S (to be inside the region) AND equal none of S (to keep the pointer null). S ∩ ¬S = ∅ => no value reaches the bug state => INFEASIBLE.
4. Conversely, if the path admits the variable and then only excludes a PROPER SUBSET of S (some member of S is still allowed), the pointer could legitimately stay null for that remaining member => do NOT conclude contradictory (could be a real bug).
5. If uncertain or the evidence is not decisive, return "insufficient_info" — never force an FP.

Output ONLY valid JSON. No markdown fences, no extra text."""

SEMANTIC_FEASIBILITY_PROMPT = """## Semantic Feasibility — Range / Finite-Domain Variable Conflict

### Bug Context
- **Type**: {bug_type}
- **Function**: {bug_function}
- **Location**: {bug_location}:{bug_line}

### Reported CSA Path Decisions (bug function)
Each line is a branch the CSA report claims the path took. `TRUE` = the condition held to stay on the path; `FALSE` = the condition was negated to stay on the path.
```
{path_decisions}
```

### Task
Using the control-flow source skeleton and the recovered finite-domain facts below, check whether the reported path forces a guard variable to be simultaneously inside and outside its pinned constant set (per the rules in the system prompt). Focus on how the deref'd pointer becomes non-null:
- If reaching the deref with the pointer NULL requires taking FALSE on every guard that would assign the pointer, while entering the deref region requires the SAME guard variable to equal one of those tags, the path is INFEASIBLE => determination `contradictory` (false positive).

### Output Format
```json
{{
  "determination": "consistent | contradictory | insufficient_info",
  "guard_variable": "h",
  "pinned_set": ["A", "B", "C"],
  "reasoning": "Explain the memberships/exclusions on the reported path and why they do or do not conflict.",
  "detected_contradiction": null
}}
```
If contradictory, set `detected_contradiction` to a one-line statement of the conflict (e.g. "h must equal one of {{rrot,PCAm,LTra,PcAm,Viqm,Pcam}} to enter the region, but is excluded from all of them to keep `lt` null"). If consistent or uncertain, leave `detected_contradiction` null."""
# ---------------------------------------------------------------------------
# Given a path segmented at control-flow boundaries, the LLM selects a
# feasible path through each segment.  Selections are then merged and
# checked for global conflicts in a separate LLM call.
# ---------------------------------------------------------------------------

SEGMENT_PATH_SELECTION_SYSTEM_PROMPT = """You are a C++ control-flow analysis expert. Given a single segment of a CSA bug report path, select a feasible execution path through that segment.

INPUT:
- **Sliced source code**: the C++ code relevant to this segment (PDG-sliced)
- **Completed constraints**: known variable constraints from constraint completion
- **Segment boundaries**: line range and function context
- **Preceding state**: the program state entering this segment (from previous segments)
- **Control events**: the branch decisions in this segment from the CSA report

TASK:
Select a feasible execution path through this segment by deciding, for each branch point, which branch is taken. Your decisions must:
1. Be consistent with the completed constraints
2. Be consistent with the preceding state
3. Reach the segment's exit point in a valid program state

If the CSA-reported branch direction is feasible, prefer it. If not, select the alternative branch.

OUTPUT: JSON with branch decisions and the resulting program state.

Output ONLY valid JSON. No markdown fences, no extra text."""

SEGMENT_PATH_SELECTION_PROMPT = """Select a feasible execution path through Segment {segment_index}/{num_segments} of the bug report path.

## Bug Context
- **Type**: {bug_type}
- **File**: {bug_location}:{bug_line}
- **Function**: {bug_function}

## Segment Information
- **Label**: {segment_label}
- **Line range**: L{line_start} - L{line_end}
- **Function**: {function_name}

## Sliced Source Code (statements relevant to this segment)
```cpp
{source_context}
```

## Completed Constraints (from constraint completion)
{completed_constraints}

## Preceding State (entering this segment)
```
{preceding_state}
```

## Control Events in This Segment
These are the branch decisions from the CSA bug report:
```
{branch_events}
```

## Global Constraints (from project-wide analysis)
{global_constraints}

## Task

For this segment, select a feasible execution path:

1. **Review** the code, constraints, and preceding state.
2. **For each branch point**, decide which branch to take:
   - If the CSA-reported direction is feasible given the constraints, prefer it.
   - If the reported direction conflicts with constraints, select the alternative.
3. **Record** the resulting program state after this segment executes.

### Output Format
```json
{{
  "segment_index": {segment_index},
  "is_feasible": true,
  "branch_decisions": [
    {{
      "line": 42,
      "branch": "true",
      "reasoning": "x > 5 is consistent with constraint x >= 10"
    }}
  ],
  "established_state": "After segment: x >= 10, y == nullptr, flag == true",
  "explanation": "All branch conditions are satisfiable given the completed constraints."
}}
```

If infeasible:
```json
{{
  "segment_index": {segment_index},
  "is_feasible": false,
  "branch_decisions": [],
  "established_state": null,
  "explanation": "No consistent path through this segment: contradictory conditions...",
  "contradiction": "L42: x > 5 conflicts with constraint x <= 0 from completion."
}}
```"""


# ---------------------------------------------------------------------------
# Prompt 12: Global Path Merge (Step 4 post-processing)
# ---------------------------------------------------------------------------
# After each segment has a selected path, merge them and check for global
# constraint conflicts across segments.
# ---------------------------------------------------------------------------

PATH_MERGE_CHECK_SYSTEM_PROMPT = """You are a C++ formal verification expert. Given selected execution paths for each segment of a CSA bug report, check whether the combined path is globally consistent.

Each segment has its own branch decisions and established state. Check whether:
1. The **accumulated state** across segments is consistent (no contradictions between what one segment establishes and another segment assumes).
2. The **global path** (all segments combined) can reach the bug trigger point without contradictions.
3. All **project-wide invariants** are respected throughout the combined path.

Output ONLY valid JSON. No markdown fences, no extra text."""

PATH_MERGE_CHECK_PROMPT = """Check whether the combined execution path (all segments merged) is globally consistent.

## Bug Context
- **Type**: {bug_type}
- **File**: {bug_location}:{bug_line}
- **Description**: {bug_description}

## Completed Constraints (from constraint completion)
{completed_constraints}

## Global Constraints (project-wide invariants)
{global_constraints}

## Selected Path Per Segment

{segment_selections}

## Task

1. **Trace the state** through all segments sequentially. Start from function entry.
2. **At each segment boundary**, verify the segment's established_state is consistent with the next segment's assumptions.
3. **Check for global contradictions**: a variable set in segment 1 may create a constraint that conflicts with a branch decision in segment 3.
4. **Verify the bug trigger point**: can the combined path actually reach the bug site?

### Output Format
```json
{{
  "is_globally_feasible": true,
  "explanation": "All segments are consistent. The combined path reaches the bug trigger with constraints: x >= 10, ptr == NULL.",
  "contradictions": [],
  "segment_states": [
    {{"segment": 0, "state": "x >= 10"}},
    {{"segment": 1, "state": "x >= 10, ptr = NULL"}}
  ]
}}
```

If global conflicts found:
```json
{{
  "is_globally_feasible": false,
  "explanation": "Segment 1 establishes ptr = non-NULL, but segment 3 requires ptr == NULL at L85.",
  "contradictions": ["Segment 1 (L40): ptr allocated; Segment 3 (L85): ptr can be nullptr"],
  "conflicting_segments": [1, 3],
  "segment_states": []
}}
```"""


# ---------------------------------------------------------------------------
# Prompt 14: State Lifecycle Analysis (Step 2.7)
# ---------------------------------------------------------------------------

STATE_LIFECYCLE_SYSTEM_PROMPT = """You are a C++ state lifecycle analysis expert. Given a function call sequence extracted from a CSA (Clang Static Analyzer) bug path and its source code, determine whether the "uninitialized variable" flagged by the CSA is actually initialized through cross-function state management that the CSA does not model.

The CSA treats function calls as mostly opaque and does not track internal state variables across call boundaries. A variable may appear "uninitialized" to the CSA because it doesn't model the lifecycle of a state parameter across multiple function calls.

You should respond ONLY with valid JSON. No markdown fences, no extra text."""

STATE_LIFECYCLE_PROMPT = """Analyze whether the Error Start variable at {bug_location}:{bug_line} is truly uninitialized, considering the LIFECYCLE of state variables across function calls.

## Bug Context
- **Type**: {bug_type}
- **File**: {bug_location}:{bug_line}
- **Function**: {bug_function}
- **Error Start Variable**: {error_start_var}

## Call Sequence (extracted from CSA path)
The following function calls appear in order in the bug report path:

{call_sequence}

## Source Code Context
```
{source_context}
```

## Known Patterns (from static analysis)
{known_patterns}

## Function Behavior Table
{function_behavior_table}

## Analysis Task

### Step 1: Identify State Variables
Look at the called functions' parameters. Which are:
- **State IN parameters**: variables that carry state INTO a function (read before write)
- **State OUT parameters**: variables that are MODIFIED BY a function (written through a pointer/ref parameter)

### Step 2: For Each Called Function, Determine Its State Behavior
Fill in: does the function SET a state variable to a known initial value?
If a "state reset" function is called BEFORE the Error Start event, the flagged variable may be initialized through the reset function's effect on a subsequent function call.

### Step 3: Determine if a State Lifecycle Guarantees Initialization
Given the call sequence and function behaviors:
1. Does the Error Start variable depend on a state parameter of a called function?
2. Is that state variable initialized by a "reset" function called earlier in the path?
3. Does a "non-reset" function (which does NOT touch the state variable) sit between the reset and the Error Start?
4. If the state is guaranteed to be in its initial value when the Error Start function is called, then the flagged read is actually initialized — it's a false positive caused by the CSA not tracking the state lifecycle.

### Output Format
```json
{{
  "state_variables": ["utf8state"],
  "reset_functions": ["wslay_event_imsg_reset"],
  "non_reset_functions": ["wslay_event_imsg_set"],
  "error_start_initialized_by_lifecycle": false,
  "lifecycle_type": "state_reset" | "loop_carried_init" | "union_type_punning" | null,
  "explanation": "Explain why the lifecycle does or does not initialize the Error Start variable.",
  "calling_sequence_summary": "imsg_reset -> ... -> decode"
}}
```"""


# ═══════════════════════════════════════════════════════════════════════════════════
# Shortest-Path + Conflict-Driven Learning prompt templates (Phase 3-4)
# ═══════════════════════════════════════════════════════════════════════════════════

PATH_CANDIDATE_SELECTION_SYSTEM_PROMPT = """\
You are a C++ control-flow analysis expert. Given a segment of a CSA bug
report path and a ranked list of pre-computed shortest path candidates,
select the most feasible candidate.

PREFER the shortest candidate (Candidate #1) unless it violates a constraint.
Only reject a candidate if it is PROVABLY infeasible given the completed
constraints and preceding state.
"""

PATH_CANDIDATE_SELECTION_PROMPT = """\
## Segment Path Selection

### Constraints
{constraints}

### Preceding State (from previous segments)
{preceding_state}

### Pre-Computed Path Candidates (ranked by shortest path: S / C / V)
{candidates}

### Task
Select the most feasible candidate. Prefer Candidate #1 unless it violates
a constraint. Reject only if PROVABLY infeasible.

Return your selection as JSON with:
- selected_candidate: int (candidate number, 1-indexed)
- reasoning: why this candidate is feasible
- established_state: variables and constraints confirmed by this selection
"""

SEGMENT_RESELECT_PROMPT = """\
## Segment {segment_index} RE-SELECTION (Round {round}/3)

### What Went Wrong Last Time
Your previous choice for this segment contributed to a global conflict.

### Avoidance Rules (Learned from Previous Rounds)
{avoidance_rules}

### Path Candidates (ranked by S→C→V)
{candidates}

### Frozen Context
{frozen_context}

### Task
Select a DIFFERENT candidate path that:
1. Avoids the conflicts described in the avoidance rules
2. Satisfies ALL completed constraints
3. Is reachable from the preceding state
4. Can reach the postcondition point

If you believe the previous choice was actually correct and the conflict
is a false alarm, explain why and re-confirm the same choice.

Return your selection as JSON with:
- selected_candidate: int
- reasoning: why this candidate resolves the conflict
- established_state: variables and constraints confirmed by this selection
"""

CONFLICT_LEARNING_SYSTEM_PROMPT = """\
You are a program analysis expert specializing in conflict diagnosis.
Given a set of inter-segment constraint conflicts in a CSA bug report path,
extract an 8-layer Hierarchical Conflict Pattern.

Layers to extract:
1-2 (Location + Constraint): Already extracted deterministically.
3 (Semantic Object): What program objects are involved? What are their roles?
4 (Semantic Type): Classify the conflict type.
5 (Root Cause): What is the minimal unsatisfiable core?
6 (Propagation Chain): How did the conflict propagate across segments?
7 (Repair Hint): What should change to resolve this?
8 (Generalization): What reusable pattern does this represent?

For Layers 3-8, reason deeply about the program semantics. Do NOT just
restate the constraint clash — explain WHY it happened in terms of the
program's dataflow and control flow.
"""

CONFLICT_LEARNING_PROMPT = """\
## Global Merge Failure

The following inter-segment conflicts were detected:

{conflict_details}

### Segment Established States
{all_segment_states}

### Violated Constraints (deterministic)
{deterministic_constraint_conflicts}

---

Extract the 8-layer Hierarchical Conflict Pattern. Output as JSON.
"""


# ---------------------------------------------------------------------------
# Prompt: TP verification — LLM simulated execution of a generated test case
# ---------------------------------------------------------------------------

SIMULATION_SYSTEM_PROMPT = """You are a precise C++ execution simulator AND a static-analysis auditor. Given a concrete C++ test case (POC) and the source of the buggy function that a static analyzer (Clang Static Analyzer) flagged, you simulate the test case's execution step by step and, crucially, check whether the CSA report's CRITICAL PATH NODES actually hold during real execution — a node that does not hold is reported against the numbered step that states it.

The CSA report path is a chain of critical nodes: its assumptions ("Assuming ..."), its branch decisions, the values it claims (e.g. "pointer value is null", "error is equal to 0", "value is garbage / uninitialized"), and finally the buggy statement (of type "{bug_type}" at {bug_file}:{bug_line} in function "{bug_function}").

For each executable statement, trace the actual control flow and data flow deterministically (respect exact branch conditions, loop bounds, API outcomes, and concrete argument values). Then check each critical path node: does that assumption / branch decision / claimed value actually hold under real execution with the test case's concrete inputs?

FIRST, AUDIT THE TEST CASE ITSELF FOR CORRECTNESS. The test case is NOT compiled by a real toolchain — you are the only verifier, so before simulating execution you must check whether it is valid and self-consistent as C++:
- Declarations / definitions: are all used types, variables, functions, and macros defined (or stubbed with a clear `// mock`)? Are argument types and arity consistent with the called functions' signatures?
- Initialization: is every value READ before being WRITTEN? An uninitialized local passed as an output parameter is a valid idiom, but reading it before a write is a real defect.
- Missing setup: does the test case actually invoke the buggy function and reach the flagged statement, or does it early-return before exercising the bug?
- Mocks: any `// mock` stub must obey the real API contract (success ⟹ non-null output; must write its output parameter). A stub that violates this is a POC DEFECT.
Report any defect as a POC DEFECT (fixable) — do NOT treat a defective test case as evidence that the bug does or does not fire; say which part of the reasoning is invalidated.

SECOND, VERIFY PATH CONFORMANCE STEP BY STEP. The report provides an ORDERED list of CSA path steps (assumptions, branch decisions, claimed values, and the final report). For EACH step, in order, state what the report expects at that step and what the POC's simulated execution actually produces there, and whether they MATCH. Judge each step by its KIND — do NOT require the POC to literally re-execute the same statement/branch:
- ASSUMPTION events ("Assuming X is non-null", "Assuming X is equal to N"): SATISFIED when the POC's concrete inputs actually make that assumption TRUE (e.g. the POC passes a non-null argument). The POC does not need to execute the same function or branch — the assumption is a property of the values, not of the trace.
- BRANCH DECISIONS ("Taking true/false branch at L…"): SATISFIED when the POC's execution takes that branch with the same outcome at that point. But a branch in a DIFFERENT function that the report merely passes through (e.g. a callee's internal test) is satisfied as long as the POC does not force the opposite outcome anywhere on its path.
- CLAIMED VALUES / KEY EVENTS ("Null pointer value stored", "Error Start", "value is garbage", API outcomes) and the final REPORT step: SATISFIED when the POC's execution actually produces that same state (the same field/pointer holds the same value) at the corresponding point.
Mark a step UNSATISFIED only when the POC's execution genuinely contradicts it — e.g. the POC manufactures a field value or object state the report's steps never assume, or it reaches the flagged statement through a state that differs from what the report claims. A non-conforming "trigger" is an artifact of the wrong test case — it is NOT a valid TP, and it is NOT evidence that the report is an FP either; it only means the test case must be regenerated to match the reported path.

Audit rules (these are the common reasons an FP report gets misclassified as TP):
- Check the CSA path for INTERNAL CONTRADICTION. A report is self-inconsistent (and therefore almost always an FP) when two of its own claims cannot hold together — e.g. it assumes "error == 0" (an API call SUCCEEDED) yet also claims that API's returned output pointer is null; or it assumes a null-guard "if (x != nullptr)" is TAKEN while also claiming x is null (reaching the deref requires the guard to be true, which requires x to be non-null).
- DO NOT trust POC stubs / mocks that violate real API contracts. If the POC fakes an API (getaddrinfo, read, memcpy, malloc, new, snprintf, ...) with an unrealistic result (e.g. a "successful" call that returns a null pointer, or fails to write its output parameter, or returns a value no real call could), treat that as a POC DEFECT, not as evidence the bug fires. Use the real API's documented postcondition instead (e.g. getaddrinfo success ⟹ the returned addrinfo list and each ai_addr are non-null).
- DO NOT trust DIRECT CONSTRUCTION of a struct that real code only ever receives from a system/dependency API. A POC that bypasses the API by hand-building the struct (e.g. a local `addrinfo ai; memset(&ai,0,sizeof ai); ai.ai_addr = nullptr;` instead of calling getaddrinfo) and fills it with a state the real API's postcondition forbids (e.g. null ai_addr, a sockaddr the API would never return) is a POC DEFECT, NOT a reachable real-world state. Check whether any REAL caller can reach the flagged function with such a struct: if every real call site goes through the API (so the invalid field can never occur), the flagged bug is unreachable in real usage → the report is an FP, and the fabricated struct is why a prior classification trusted it.
- Audit HOW the test case reaches the flagged statement, not just whether the statement dereferences null. A true positive must reach it through a REAL public caller with real inputs. Treat each of these as a manufactured entry (a state/literal no real caller can produce) → the bug is UNREACHABLE in real usage → FP, NOT a trigger:
  (a) the test calls the buggy INTERNAL/leaf/template function DIRECTLY with a fabricated null/contradictory argument (e.g. `heap_heapify(..., nullptr, ...)` or `scanner->scan_codes(..., ids=nullptr, ...)`), instead of driving the real public method that reaches it with the buffers a real caller passes (which are non-null);
  (b) the test CONSTRUCTS the buggy object/container in an empty/zero-size degenerate state whose internal pointer is null (e.g. a zero-size allocator) and then operates on that null, when every real caller sizes it > 0 before the operation;
  (c) the test DEFEATS VIRTUAL DISPATCH by defining a subclass that omits an override so an overridable BASE-class method runs, when every real subclass overrides that method (so the base default is never the actual dispatch target for a real object);
  (d) the test RE-IMPLEMENTS/STUBS the real library function (e.g. under `namespace {project}`) with a made-up internal format/magic, instead of calling the REAL function on a REAL input it can parse.
  In all four cases the "trigger" is an artifact of the manufactured POC entry — say which of (a)–(d) applies and conclude FP.
- NULL-POINTER-ARGUMENT UB — do NOT excuse it by the length. Passing a null pointer to an API whose parameter requires a valid pointer (a `nonnull`-annotated parameter such as `memcpy`/`memmove`/`memset`/`memcmp`, or any function that dereferences it) is a contract violation and C++ undefined behavior. The value of the count/length argument is IRRELEVANT: a length of 0 (e.g. `memcpy(nullptr, nullptr, 0)`, `memset(nullptr, 0, 0)`) does NOT make the call well-defined, does NOT make it a harmless no-op, and must NOT be used to excuse the defect. Do not reason "the size is zero so it cannot crash, therefore FP" — that reasoning is wrong for this bug class.
- A nonnull-annotated parameter (e.g. `__attribute__((nonnull))`) receiving a null pointer is a real bug whenever the null can actually arise from a real caller (not a direct hostile call). The null does NOT need to be "used for an actual access": the contract violation itself is the defect, and a zero-length operation does NOT make it safe (`memcpy(NULL, NULL, 0)` is still UB).
- Track CROSS-PROCEDURE writes. A value the analyzer flags as "uninitialized / garbage / undefined" may have been written by the CALLER or a CALLEE before the flagged read. If any code path writes it before the flagged use, it is NOT uninitialized at that use.
- Recognize OUTPUT-PARAMETER patterns. Passing the address of an uninitialized local as an output parameter (a function that writes *out before reading it) is a valid, common C++ idiom, NOT a bug by itself. Only flag it if the value is actually READ before any write.
- Distinguish "the POC is wrong / misses a precondition" (fixable; reach the bug with a better test) from "the path is genuinely impossible" (the analyzer's claim is false → the report is a False Positive).
- Distinguish TP from FP explicitly. A TP requires the buggy statement to be GENUINELY reached AND the triggering condition to hold in real execution. If the flagged statement cannot be reached, or is reached but the operation is actually safe, or the analyzer's claim is internally contradictory / contradicted by real API or cross-procedure semantics, the report is an FP — and explain WHY an FP could have been misclassified as TP (e.g. the analyzer could not see a caller's write, did not inline a guard / IsShared() / helper, or a prior classification trusted an unrealistic POC mock).
- Be conservative and factual: only claim the bug fires if the exact buggy statement is actually reached with the conditions that trigger it.
- Respond ONLY with valid JSON, no markdown fences, no extra text."""

# Memory-leak-specific simulation semantics.  Appended to the SIMULATION user prompt ONLY when the
# report's bug_type is a memory leak (see SimulationVerifier.simulate_execution).  The default
# `triggered`/`fp_likelihood` rules elsewhere are written in null-deref / crash terms, which do NOT
# transfer to a leak: "reaches the allocation site" is NOT a fired leak.  A leak fires only when the
# allocated object is genuinely ABANDONED (never freed AND never transferred to a live owner).
MEMORY_LEAK_SIMULATION_RULES = """### MEMORY-LEAK SEMANTICS (this report is a memory leak — read before answering)
For a memory-leak report, `triggered` is NOT defined by "the buggy line is reached".  A leak fires
only when ALL THREE hold on the simulated path:
  1. ALLOC: an object is allocated at the CSA allocation site (the `new`/malloc the report names);
  2. ABANDON: control then leaves the owning scope on that path — via `return`, an exception/throw,
     or a fall-through that exits the function WITHOUT storing or freeing the object;
  3. UNOWNED: the object is never freed (`delete`, `unique_ptr`, RAII destructor, a `_free`/cleanup
     helper) AND never transferred to a live owner (assigned to the return value, handed to an
     owning container, or an `own_fields = true` style transfer) BEFORE that abandonment.
Reaching the allocation with the object later stored into the return value or an owner is NOT a
leak — it is a normal successful construction.  So `triggered` for a leak ≈ whether the object is
actually abandoned-and-unowned on a real path, NOT whether the allocation line is reached.

COMMON MEMORY-LEAK FALSE POSITIVES (call these FP with evidence, do NOT force a trigger):
- STORED-TO-OWNER ON WELL-FORMED PATHS: on every well-formed / in-contract continuation of the
  flagged path, the allocated object is stored into the return value or an owning container before
  the function exits (e.g. `idx = idxnsg;` then `return idx;`), so no well-formed real caller ever
  leaks it.
- BROKEN-INPUT-ONLY ABANDON: the object has no cleanup, but it leaks ONLY because a later
  read/parse call throws on a corrupt / truncated / out-of-contract input (e.g. a mid-stream
  `FAISS_THROW_IF_NOT_FMT` when a read returns fewer bytes than expected).  No well-formed file or
  valid caller can reach that throw, so the leak is unreachable by any real, well-formed caller →
  the report is an FP.  Say exactly which read can throw and that a well-formed input would never
  take it.
- FREED / OWNED ANYWHERE: the object is freed on any path (`delete`, RAII, `unique_ptr`, a cleanup
  helper) or transferred to a live owner (returned, `push_back`ed, `own_fields` set) before the
  function exits → it is not leaked on that path.
- SELF-CONTRADICTORY CSA PATH: the branch that allocates the object and the branch/state under which
  CSA claims it leaks are mutually exclusive (cannot both hold) → the report is internally
  inconsistent → FP.
A leak is a real TP only if a genuine well-formed caller abandons the object unowned (e.g. a normal
early return that skips the free with no owner transfer, or a genuinely reachable failure on
in-contract input).

For THIS memory-leak report, ALSO emit the extra JSON field `leak_chain` (the code reads it to reach
a decisive verdict, so it MUST be present and non-null — do not omit it):
    "leak_chain": {
        "alloc_line": <int line of the allocation site CSA names>,
        "alloc_expr": "<the `new`/allocation expression, e.g. `new faiss::IndexNSGPQ()`>",
        "wellformed_continuation_stores_to": "return" | "owner" | null,   // where the object goes on every WELL-FORMED continuation (return value, owning container/field)
        "freed_on_any_path": <bool>,   // is the object freed (delete/RAII/unique_ptr/cleanup helper) on any path before the function exits?
        "abandon_path": "exception_after_alloc" | "manual_no_free_early_return" | null,   // how the object COULD be abandoned, if at all
        "requires_broken_input": <bool>,   // does that abandon need corrupt/truncated/out-of-contract input to occur?
        "leak_fires_on_wellformed": <bool>   // AUTHORITATIVE: is the object genuinely abandoned-UNOWNED on a real WELL-FORMED caller path? (only this makes it a real TP leak)
    }
Be conservative on `leak_fires_on_wellformed`: it is true ONLY when a genuine well-formed caller can
reach the abandon without the object being freed or handed to an owner."""


SIMULATION_PROMPT = """Audit the following test case for correctness, then simulate its execution, verify the CSA report's path step by step, and determine whether the flagged bug is a true positive (TP) or a false positive (FP).

IMPORTANT: The test case is NOT compiled by a real toolchain. YOU are its only verifier. Before simulating, first judge whether the test case is correct and self-consistent as C++: are all used symbols/initializations present, do the API calls match their signatures, is every value initialized before read, and does it actually reach the flagged statement? Report any defect explicitly via the output fields below; do not let a defective test case distort the TP/FP conclusion.

## Bug Information
- **Type**: {bug_type}
- **Location**: {bug_file}:{bug_line}
- **Function**: {bug_function}

## CSA Critical Path Nodes (the analyzer's assumptions / branch decisions / claimed values leading to the bug)
{completed_path_summary}

For EACH critical node above, determine whether it actually holds under real execution with the test case's concrete inputs. A node that one of the numbered path steps states (they name the same source line) is reported against that step in Task 3; a node no step states is reported in `reason`; two of the report's own claims that cannot hold together go in `csa_contradiction`. Watch specifically for:
- internal contradictions in the path (e.g. an API assumed successful yet its output claimed null; a null-guard assumed taken yet the pointer claimed null);
- POC stubs/mocks that contradict real API contracts (successful call returning null / not writing its out-param);
- DIRECT CONSTRUCTION of a struct that real code only receives from a system/dependency API, filled with a state that API's postcondition forbids (e.g. a hand-built `addrinfo` with `ai_addr=nullptr`). If no real caller can reach the flagged function with such a struct, the bug is unreachable in real usage → FP;
- MANUFACTURED ENTRY — the test reaches the flagged statement NOT through a real public caller but by a shortcut that no real caller can produce. If ANY of these apply, the null is fabricated and the bug is unreachable in real usage → FP, do NOT count it as a trigger:
  (a) it calls an internal/leaf/template function directly with a fabricated null/contradictory argument (e.g. `heap_heapify(..., nullptr, ...)`, `scanner->scan_codes(..., ids=nullptr, ...)`) instead of the real public method that passes valid buffers;
  (b) it builds the buggy object in an empty/zero-size state with a null internal pointer (every real caller sizes it > 0 before the flagged operation);
  (c) it defines a subclass that drops a virtual override so a base-class default runs, when every real subclass overrides it (base never actually dispatched for a real object);
  (d) it re-implements/stubs the real library function with a made-up internal format/magic instead of calling the real function on real input;
- NULL-POINTER-ARGUMENT UB — do NOT excuse it by length. Passing a null pointer to a `nonnull`-annotated / pointer-requiring API (e.g. `memcpy`, `memmove`, `memset`, `memcmp`) is a contract violation and undefined behavior REGARDLESS of the count/length: a zero length (e.g. `memcpy(nullptr, nullptr, 0)`) does NOT make it well-defined or harmless. If the null can arise from a real caller, the call IS the reported defect → TP (do not argue "size 0 ⇒ no crash ⇒ FP");
- values the analyzer calls "uninitialized / garbage" that are actually written by a caller or callee before use;
- output-parameter patterns (passing &of_uninitialized_local as an out-param is valid, not a bug).

## CSA Report Path Steps (ordered — the POC's execution MUST reproduce each of these)
{csa_path_steps}

For EACH step above, verify by simulation that the POC produces the SAME result the report expects at that step. Judge by the step's kind: an ASSUMPTION step is satisfied when the POC's concrete inputs make the assumption TRUE (no need to re-execute the same function/branch); a BRANCH step is satisfied when the POC takes that outcome (a branch in another function the report merely passes through is fine unless the POC forces the opposite); a CLAIMED-VALUE / KEY-EVENT / REPORT step is satisfied when the POC actually produces that same state. Mark a step UNSATISFIED only when the POC genuinely contradicts it. Any mismatch ⇒ the test case is wrong (`path_conformance_ok=false`). Each step's verdict becomes either silence (satisfied) or its own `path_mismatches` row (unsatisfied) — see Task 3; no prose per step.

## Concrete Preconditions / Values
{preconditions}

## Test Case (POC) to Simulate
```cpp
{poc_code}
```

## Source of the Buggy Function
```cpp
{source_context}
```

## Task
1. Audit the test case for correctness (symbols, types, init-before-read, API signatures, reaching the buggy statement; flag any mock that violates a real API contract). Summarize issues in `test_case_issues`.
2. Simulate the test case execution step by step with its concrete inputs.
3. PATH CONFORMANCE: walk the ordered CSA report path steps above and check EACH one — they are numbered `1..N`. Report the failures ONLY; do NOT echo the step text back (it is already above) and do NOT write a record for a satisfied step:
   - `steps_checked`: how many numbered steps you walked — MUST equal N, the number of the last step above;
   - `path_mismatches`: one record per UNSATISFIED step — `{{"step": <n>, "poc_result": "<what the POC's execution actually produces there>", "why": "<how that differs from the report's expectation>"}}` — and nothing for the satisfied ones (no empty placeholder records). Every satisfied step is implicit. List at most 10 records: when the POC diverges from the reported path at more steps than that (e.g. it takes a different top-level branch and every downstream step differs), give the 10 that best explain the divergence — the step where the paths first differ, the branch/claimed-value steps around it, and the reported step — and describe the rest as a whole in `reason`. Never pad the list with rows that do not actually differ.
   If `path_mismatches` is empty, every step conforms and `path_conformance_ok=true`; if it is non-empty, set `path_conformance_ok=false` — the POC does not reproduce the reported path (regenerate it), which is NOT by itself an FP. Never write `path_conformance_ok=false` with an empty `path_mismatches`.
4. Decide the report's true nature: TP (genuinely reachable and triggers) or FP (path impossible, self-contradictory, or the flagged condition is actually safe).
5. If FP, explain WHY the analyzer (or a prior TP classification) could have mistaken it for TP.

### Definition of `triggered` (read before answering)
- `triggered` MUST be **true only if the flagged bug actually fires during the simulated execution**: e.g. for a null-dereference, the dereferenced pointer is **actually NULL/invalid at the deref site and the deref crashes**.
- For a report about a null pointer passed to a `nonnull`-annotated / pointer-requiring API (bug type like "Argument with 'nonnull' attribute passed null"), `triggered` is **true** as soon as the POC actually passes a null pointer to that API on a path reachable from a real caller. The count/length being 0 does **NOT** make it a non-trigger and the call does **NOT** need to crash — `memcpy(nullptr, nullptr, 0)` is UB and counts as triggered. Never set `fp_likelihood=definite_fp` merely because "the size is zero so nothing is accessed".
- **Reaching the buggy line with a valid/non-null pointer is NOT a trigger.** If the buggy statement executes but the dereference is safe (e.g. `ai_addr` is a concrete non-null object), set `triggered=false`.
- If the POC does **not** set up the null/invalid condition that the CSA path requires to fire (it passes a valid object where the bug needs a null pointer), the POC **cannot trigger** — say so honestly, `triggered=false`.
- `triggered` and `fp_likelihood` MUST be mutually consistent: never set `triggered=true` together with `fp_likelihood=definite_fp`. If the flagged condition is actually safe (deref target is concrete non-null, or a real-API success postcondition forbids null), set `triggered=false` and `fp_likelihood=definite_fp`.
- `path_conformance_ok` MUST be **false whenever any report step's expected result is not reproduced by the POC's execution**, even if the bug appears to fire. In that case the POC is a test-case defect to be regenerated — do NOT report it as a TP, and do NOT report it as an FP. It MUST also be consistent with the mismatch list: `path_conformance_ok = true` iff `path_mismatches` is empty.

Output JSON:
{{
  "triggered": true,
  "test_case_correct": true,
  "test_case_issues": ["Any defect in the test case (undefined symbol, missing init, signature mismatch, unrealistic mock, early return before the bug, ...). If the test case is correct, this is an empty list."],
  "steps_checked": 92,
  "path_mismatches": [{{"step": 47, "poc_result": "<what the POC's execution produces at that step>", "why": "<how that differs from the report's expectation>"}}],
  "path_conformance_ok": true,
  "trace": ["L<line>: <statement> → <variable state>", "..."],
  "reason": "How/why the bug does or does not fire, including which CSA critical node(s) are violated if FP.",
  "csa_contradiction": "If the CSA path is internally self-contradictory, describe the contradiction; otherwise null.",
  "fp_likelihood": "definite_fp | likely_fp | uncertain | tp",
  "blocking_reason": "If NOT triggered: the exact condition/precondition/input that prevented reaching or triggering the bug. If triggered, null.",
  "suggested_input_change": "If NOT triggered due to wrong POC setup: a concrete change to inputs/setup that would reach the buggy path. If triggered or unreachable, null."
}}

Field sizes (keep the reply compact — it must fit the token budget in one piece):
- `steps_checked` + `path_mismatches` replace the old per-step record list. There is no `report_expectation` field to fill (the numbered steps above are the expectation) and no per-step `note` for satisfied steps; satisfied steps cost nothing at all, and `path_mismatches` is capped at 10 records however many steps differ.
- `trace`: emit it ONLY when `triggered` is false (it explains why nothing fired), at most 10 entries, one per line ON the reported path — no full variable-state dumps. When `triggered` is true, use `[]`."""


SIMULATION_REFINE_SYSTEM_PROMPT = """You are a C++ engineer. A static-analysis bug was flagged as a true positive (TP), and a generated test case (POC) was supposed to reproduce it. You simulated executing the POC and it did NOT trigger the bug.

Given the POC, the simulation trace showing why it did not trigger, and the blocking reason, produce a REFINED POC that actually reaches and triggers the flagged bug.

Rules:
- Fix the root cause of non-triggering: correct inputs, preconditions, API usage, loop setup, or the calling context so the execution path reaches the buggy statement under the triggering condition.
- REPRODUCE THE REPORTED PATH: the refined POC must make the real code produce the same state the CSA report's path steps claim at each step. Never manufacture a state (direct writes to internal fields, hand-built dependency structs, re-implemented/stubbed library types) that the report's own path does not assume — such a "trigger" is invalid and will be rejected.
- Preserve the intent (exercise the exact buggy function at the exact file location).
- Do NOT weaken the target function or alter its semantics.
- If the bug genuinely cannot be reached (path is unreachable regardless of setup), output the ORIGINAL POC unchanged with no fabrication.
- Output ONLY valid C++ code, no explanations, no markdown fences."""

SIMULATION_REFINE_PROMPT = """Refine the following POC so that simulated execution actually triggers the flagged bug.

## Bug Information
- **Type**: {bug_type}
- **Location**: {bug_file}:{bug_line}
- **Function**: {bug_function}

## Original POC (did NOT trigger)
```cpp
{poc_code}
```

## Simulation Result of the Original POC
- triggered: {triggered}
- reason: {reason}
- blocking_reason: {blocking_reason}
- suggested_input_change: {suggested_input_change}

## Simulation Trace
{trace}

## Path-Conformance Failures (the previous POC did NOT reproduce these CSA report steps)
{path_conformance}

## Source of the Buggy Function
```cpp
{source_context}
```

## Task
Output a refined POC that reaches and triggers the bug WHILE reproducing the CSA report's path.
Only output C++ code.

Refinement constraints:
- Do NOT fabricate an unrealistic state to force the trigger. In particular, do NOT hand-construct a struct that real code only receives from a system/dependency API (e.g. an `addrinfo` with `ai_addr=nullptr`) unless you are certain a real caller can produce it. If the only way to reach the flagged statement is through such a fabricated state that real callers cannot produce, the bug is a False Positive — keep the POC as-is rather than fabricate.
- FIX EVERY path-conformance failure listed above: the refined POC must drive the real public API so that each report step's expected state (branch decision, claimed value, object state) is actually produced by the real code, NOT manufactured by hand (no direct field writes that no real API performs, no stubbed/re-implemented library types, no bypassing the real caller). Removing a manufactured state is the correct fix even if it means the POC no longer crashes.
"""
