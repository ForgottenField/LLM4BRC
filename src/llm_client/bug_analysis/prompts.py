"""Prompt templates and few-shot examples for bug report analysis."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a bug report analysis assistant. Your task is to parse static analysis tool outputs and extract structured bug information.

## Instructions

1. Analyze the bug report content provided by the user.
2. Identify ALL distinct bug findings in the report.
3. For each finding, extract all possible fields described in the output schema below.
4. Output ONLY valid JSON matching the schema — no explanations, no markdown formatting.

## Output Schema

{
  "findings": [
    {
      "bug_type": "<CWE identifier or short name, e.g. 'CWE-79: XSS' or 'buffer-overflow'>",
      "severity": "<one of: critical, high, medium, low, informational>",
      "affected_location": {
        "file": "<file path>",
        "line": <integer line number or null>,
        "end_line": <integer or null>,
        "column": <integer or null>,
        "end_column": <integer or null>,
        "snippet": "<relevant code snippet or null>"
      },
      "description": "<2-3 sentence summary of the bug>",
      "root_cause": "<brief explanation of the underlying programming error>",
      "suggested_fix": "<high-level remediation guidance or null>",
      "poc_viability": {
        "rating": "<one of: high, medium, low, none — how viable is POC generation>",
        "exploit_preconditions": ["<list of conditions required for exploitation>"],
        "impact": "<potential impact if exploited or null>",
        "viability_reasoning": "<brief justification for the rating or null>"
      }
    }
  ]
}

## Rules

- If the report contains NO bugs, return a single finding with bug_type "NO_BUGS_FOUND" and severity "informational".
- If you cannot determine a field's value, set it to null (never fabricate data).
- For severity, map the report's severity to: critical, high, medium, low, or informational.
- For bug_type, use a CWE identifier if the tool provides one or a known mapping exists; otherwise use a clear descriptive name.
- Extract ALL findings from the report — do not skip any.
- The output must be a single JSON object with a "findings" key containing an array.
"""

FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    # ------------------------------------------------------------------ #
    # Example 1: SARIF input
    # ------------------------------------------------------------------ #
    {
        "role": "user",
        "content": """{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [{
    "tool": { "driver": { "name": "ESLint" } },
    "results": [
      {
        "ruleId": "no-unused-vars",
        "level": "warning",
        "message": { "text": "'x' is assigned a value but never used." },
        "locations": [{
          "physicalLocation": {
            "artifactLocation": { "uri": "src/app.js" },
            "region": { "startLine": 42, "startColumn": 7 }
          }
        }]
      }
    ]
  }]
}""",
    },
    {
        "role": "assistant",
        "content": """{
  "findings": [
    {
      "bug_type": "unused-variable",
      "severity": "low",
      "affected_location": {
        "file": "src/app.js",
        "line": 42,
        "end_line": null,
        "column": 7,
        "end_column": null,
        "snippet": null
      },
      "description": "Variable 'x' is assigned a value on line 42 but is never used anywhere in the file. Unused variables increase code complexity and may indicate dead code or incomplete refactoring.",
      "root_cause": "The variable was declared and assigned but never referenced after assignment.",
      "suggested_fix": "Remove the unused variable declaration, or use it if the assignment was intended for a side effect.",
      "poc_viability": {
        "rating": "none",
        "exploit_preconditions": [],
        "impact": null,
        "viability_reasoning": "Unused variables are code-quality issues, not security vulnerabilities. They cannot be exploited to cause runtime harm."
      }
    }
  ]
}""",
    },
    # ------------------------------------------------------------------ #
    # Example 2: Plain text (GCC/clang style)
    # ------------------------------------------------------------------ #
    {
        "role": "user",
        "content": """src/db.c:201:10: warning: format string is not a string literal (potentially insecure) [-Wformat-security]
    printf(user_input);
          ^~~~~~~~~~~
src/db.c:87:5: error: buffer overflow [WARN_BOF]
    strcpy(dst, src);
    ^~~~~~~~~~~~~~~~
The strcpy() function does not check buffer bounds. An attacker could overflow the destination buffer.""",
    },
    {
        "role": "assistant",
        "content": """{
  "findings": [
    {
      "bug_type": "CWE-134: Use of Externally-Controlled Format String",
      "severity": "high",
      "affected_location": {
        "file": "src/db.c",
        "line": 201,
        "end_line": null,
        "column": 10,
        "end_column": null,
        "snippet": "printf(user_input);"
      },
      "description": "The printf() call on line 201 uses user-controlled input as the format string. An attacker can provide format specifiers (%s, %x, %n) to read or write arbitrary memory.",
      "root_cause": "User input is passed directly as the format string argument to printf() instead of using a fixed format string like printf(\"%s\", user_input).",
      "suggested_fix": "Replace printf(user_input) with printf(\"%s\", user_input) or puts(user_input) to treat the input as data, not a format string.",
      "poc_viability": {
        "rating": "high",
        "exploit_preconditions": ["Attacker controls the format string argument"],
        "impact": "Arbitrary memory read and write via %n and %s format specifiers, potentially leading to code execution",
        "viability_reasoning": "Format string vulnerabilities are well-studied with reliable exploitation techniques. The %n specifier allows writing arbitrary values to arbitrary addresses."
      }
    },
    {
      "bug_type": "CWE-120: Buffer Overflow",
      "severity": "critical",
      "affected_location": {
        "file": "src/db.c",
        "line": 87,
        "end_line": null,
        "column": 5,
        "end_column": null,
        "snippet": "strcpy(dst, src);"
      },
      "description": "The strcpy() function on line 87 copies data from 'src' to 'dst' without checking that the destination buffer has sufficient capacity. This can cause a buffer overflow leading to memory corruption.",
      "root_cause": "Use of strcpy() instead of a bounded alternative like strncpy() or snprintf(). The destination buffer size is not verified before the copy.",
      "suggested_fix": "Replace strcpy(dst, src) with strncpy(dst, src, sizeof(dst)) or use snprintf(dst, sizeof(dst), \"%s\", src) to enforce bounds checking.",
      "poc_viability": {
        "rating": "high",
        "exploit_preconditions": ["Source buffer 'src' contains attacker-controlled data", "Destination buffer 'dst' is on the stack (for ROP/ret2libc)"],
        "impact": "Stack buffer overflow enabling arbitrary code execution via return address overwrite",
        "viability_reasoning": "Classic stack-based buffer overflow with mature exploitation primitives (ROP, ret2libc). The unchecked strcpy is a textbook exploitable pattern."
      }
    }
  ]
}""",
    },
    # ------------------------------------------------------------------ #
    # Example 3: Non-SARIF JSON (custom tool output)
    # ------------------------------------------------------------------ #
    {
        "role": "user",
        "content": """{
  "tool": "my-custom-scanner-v2",
  "issues": [
    {
      "id": "SQLI-001",
      "type": "sql_injection",
      "file": "src/api/users.py",
      "line": 55,
      "message": "User input concatenated into SQL query."
    }
  ]
}""",
    },
    {
        "role": "assistant",
        "content": """{
  "findings": [
    {
      "bug_type": "CWE-89: SQL Injection",
      "severity": "critical",
      "affected_location": {
        "file": "src/api/users.py",
        "line": 55,
        "end_line": null,
        "column": null,
        "end_column": null,
        "snippet": null
      },
      "description": "A SQL injection vulnerability exists at line 55 of src/api/users.py where user-controlled input is concatenated directly into a SQL query string. An attacker can craft input that modifies the query's logic or executes arbitrary SQL commands.",
      "root_cause": "Building SQL queries through string concatenation with unsanitized user input instead of using parameterized queries or an ORM.",
      "suggested_fix": "Replace string concatenation with parameterized queries (e.g., using ? placeholders in the SQL string and passing user input as query parameters).",
      "poc_viability": {
        "rating": "high",
        "exploit_preconditions": ["The database user has sufficient privileges for data extraction or modification", "The vulnerable endpoint is network-accessible"],
        "impact": "Unauthorized data access, data exfiltration, or database corruption via crafted SQL payloads",
        "viability_reasoning": "SQL injection is one of the most well-documented vulnerability classes with automated tools (sqlmap) and reliable exploitation techniques for data extraction."
      }
    }
  ]
}""",
    },
    # ------------------------------------------------------------------ #
    # Example 4: POC viability assessment (exploitable vs informational)
    # ------------------------------------------------------------------ #
    {
        "role": "user",
        "content": """src/parser.c:320:5: warning: use-after-free [WARN_UAF]
    free(ptr);
    // ... 10 lines ...
    printf("%s", ptr->name);
    ^~~~~~~~~
Pointer 'ptr' is used after it has been freed on line 320.""",
    },
    {
        "role": "assistant",
        "content": """{
  "findings": [
    {
      "bug_type": "CWE-416: Use-After-Free",
      "severity": "high",
      "affected_location": {
        "file": "src/parser.c",
        "line": 330,
        "end_line": null,
        "column": 5,
        "end_column": null,
        "snippet": "printf(\"%s\", ptr->name);"
      },
      "description": "Pointer 'ptr' is freed on line 320 but dereferenced on line 330 without being reassigned. This use-after-free can lead to arbitrary code execution if an attacker controls the freed memory region.",
      "root_cause": "The pointer is not set to NULL after free, and the code continues to reference the dangling pointer without checking its validity.",
      "suggested_fix": "Set ptr = NULL immediately after free(ptr), or restructure the control flow to avoid accessing freed memory.",
      "poc_viability": {
        "rating": "high",
        "exploit_preconditions": ["Attacker must be able to influence heap allocations between free and re-use"],
        "impact": "Arbitrary code execution or information disclosure via heap manipulation",
        "viability_reasoning": "Use-after-free in C is a well-understood vulnerability class with mature exploitation techniques (heap feng shui, tcache poisoning). The distance between free and use (10 lines) provides a realistic window for heap manipulation."
      }
    }
  ]
}""",
    },
]


def build_system_prompt() -> str:
    """Return the system instruction prompt."""
    return SYSTEM_PROMPT


def build_few_shot_examples() -> list[dict[str, str]]:
    """Return the list of few-shot message pairs."""
    return FEW_SHOT_EXAMPLES


def build_messages(report_content: str) -> list[dict[str, str]]:
    """Assemble the complete message list for the LLM call.

    Structure: system prompt → few-shot examples → user report.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": report_content})
    return messages
