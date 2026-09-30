"""Fixtures for the report-frontend tests.

Two kinds of input, on purpose:

* the *real* CSA report that lives in the repository
  (``tests/test_fp_analysis/fixtures/``) — the frontend's job is to be
  indistinguishable from the parser it replaced, and only a real report says
  whether that holds;
* a *synthetic* SARIF document, built here rather than checked in, because the
  properties worth testing (a rule table, several results, a code flow, a
  ``%SRCROOT%``-relative URI) are exactly the ones a hand-written document can
  state precisely — and because a 60-finding real export would test the file
  size, not the adapter.

The source tree the SARIF fixture points into is also synthetic
(:func:`source_tree`): resolution strategy is what the code model is for, and a
tree built to a known shape can assert *which* strategy matched, which no
corpus file can.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FP_FIXTURES = REPO_ROOT / "tests" / "test_fp_analysis" / "fixtures"

#: The CSA report the legacy parser tests use — real metadata, a real path.
CSA_REPORT = FP_FIXTURES / "report-MemoryTest.cpp-Destroy-2-1.html"

SARIF_RULES = [
    {
        "id": "cpp/uncontrolled-allocation-size",
        "name": "cpp/uncontrolled-allocation-size",
        "shortDescription": {"text": "Uncontrolled allocation size"},
        "fullDescription": {"text": "Allocating memory with a user-controlled size."},
        "defaultConfiguration": {"level": "warning"},
        "properties": {
            "tags": ["security", "external/cwe/cwe-190"],
            "security-severity": "8.1",
        },
        "helpUri": "https://codeql.github.com/codeql-query-help/cpp/"
                   "cpp-uncontrolled-allocation-size/",
    },
    {
        "id": "cpp/leap-year",
        "shortDescription": {"text": "Leap year check"},
        "defaultConfiguration": {"level": "note"},
        "properties": {"tags": ["correctness"]},
    },
]


@pytest.fixture
def csa_report() -> Path:
    if not CSA_REPORT.exists():
        pytest.skip(f"fixture missing: {CSA_REPORT}")
    return CSA_REPORT


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    """A miniature ``faiss``-shaped tree for resolution tests.

    Two files share the basename ``index_read.cpp`` under different
    directories — that is the trap the code model exists to avoid
    (``rglob(name)[0]``), and it is what the suffix strategy is asserted on.
    """
    root = tmp_path / "faiss"
    (root / "faiss" / "impl").mkdir(parents=True)
    (root / "faiss" / "invlists").mkdir(parents=True)
    (root / "faiss" / "utils").mkdir(parents=True)
    (root / "CMakeLists.txt").write_text("project(faiss)\n")

    body = "\n".join(f"// line {i}" for i in range(1, 41))
    (root / "faiss" / "impl" / "index_read.cpp").write_text(
        "// impl/index_read.cpp\n"
        + "\n".join(f"int impl_line_{i};" for i in range(2, 31)) + "\n"
        + "void fourcc_inv_printable();  // line 31\n"
        + body + "\n"
    )
    (root / "faiss" / "invlists" / "index_read.cpp").write_text(
        "// invlists/index_read.cpp\n"
        + "\n".join(f"int invlists_line_{i};" for i in range(2, 25)) + "\n"
    )
    (root / "faiss" / "utils" / "AlignedTable.h").write_text(
        "\n".join(f"// aligned {i}" for i in range(1, 71)) + "\n"
    )
    return root


@pytest.fixture
def sarif_path(tmp_path: Path) -> Path:
    """A SARIF 2.1.0 document with two findings and one code flow."""
    doc = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "CodeQL",
                        "version": "2.16.1",
                        "informationUri": "https://codeql.github.com",
                        "rules": SARIF_RULES,
                    }
                },
                "originalUriBaseIds": {"%SRCROOT%": {"uri": "file:///build/faiss/"}},
                "artifacts": [
                    {"location": {"uri": "faiss/impl/index_read.cpp",
                                  "uriBaseId": "%SRCROOT%"}},
                    {"location": {"uri": "faiss/utils/AlignedTable.h",
                                  "uriBaseId": "%SRCROOT%"}},
                ],
                "results": [
                    {
                        "ruleId": "cpp/uncontrolled-allocation-size",
                        "ruleIndex": 0,
                        "message": {"text": "This allocation size is derived from "
                                             "an unvalidated value."},
                        "locations": [{
                            "physicalLocation": {
                                "artifactLocation": {
                                    "uri": "faiss/impl/index_read.cpp",
                                    "uriBaseId": "%SRCROOT%",
                                },
                                "region": {"startLine": 12, "startColumn": 7,
                                           "endLine": 12, "endColumn": 31},
                            }
                        }],
                        "partialFingerprints": {
                            "primaryLocationLineHash": "abc123:1"
                        },
                        "codeFlows": [{
                            "threadFlows": [{
                                "locations": [
                                    {"location": {"physicalLocation": {
                                        "artifactLocation": {
                                            "uri": "faiss/impl/index_read.cpp",
                                            "uriBaseId": "%SRCROOT%"},
                                        "region": {"startLine": 5},
                                    }, "message": {"text": "value originates here"}}},
                                    {"location": {"physicalLocation": {
                                        "artifactLocation": {
                                            "uri": "faiss/impl/index_read.cpp",
                                            "uriBaseId": "%SRCROOT%"},
                                        "region": {"startLine": 9},
                                    }, "message": {"text": "passed into fourcc"}}},
                                    {"location": {"physicalLocation": {
                                        "artifactLocation": {
                                            "uri": "faiss/impl/index_read.cpp",
                                            "uriBaseId": "%SRCROOT%"},
                                        "region": {"startLine": 12},
                                    }, "message": {"text": "allocated here"}}},
                                ]
                            }]
                        }],
                    },
                    {
                        "ruleId": "cpp/leap-year",
                        "ruleIndex": 1,
                        "message": {"text": "Leap year logic looks wrong."},
                        "locations": [{
                            "physicalLocation": {
                                "artifactLocation": {
                                    "uri": "faiss/utils/AlignedTable.h",
                                    "uriBaseId": "%SRCROOT%",
                                },
                                "region": {"startLine": 66, "startColumn": 9},
                            }
                        }],
                        "suppressions": [{"kind": "inSource"}],
                    },
                ],
            }
        ],
    }
    path = tmp_path / "faiss.sarif"
    path.write_text(json.dumps(doc, indent=2))
    return path


@pytest.fixture
def infer_sarif_path(tmp_path: Path) -> Path:
    """The same container, a different producer — detection must not care."""
    doc = {
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "Infer", "version": "1.1.0", "rules": []}},
            "results": [{
                "ruleId": "NULL_DEREFERENCE",
                "message": {"text": "pointer `p` last assigned on line 3 could be null"},
                "locations": [{"physicalLocation": {
                    "artifactLocation": {"uri": "src/main.c"},
                    "region": {"startLine": 7, "startColumn": 3},
                }}],
            }],
        }],
    }
    path = tmp_path / "infer-out.sarif"
    path.write_text(json.dumps(doc))
    return path
