"""Deterministic tests for guard-pinned / finite-domain variable detection.

Never calls an LLM.  Guards the Phase-1 semantic-constraint supplement
(``domain_facts.py``): constant folding (fourcc + int), single-equality parsing,
per-variable grouping, low-noise filtering, and end-to-end extraction over the real
``pdg_faiss.json`` read_VectorTransform leaves.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_client.fp_analysis.domain_facts import (
    domain_facts_from_lines,
    extract_domain_facts,
    fold_rhs,
    format_domain_facts,
    group_domain_facts,
    _is_macro_noise,
    iter_foldable_equalities,
    parse_equality_leaf,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PDG_FAISS = REPO_ROOT / "pdg_faiss.json"


# ── fold_rhs ─────────────────────────────────────────────────────────────

def test_fold_fourcc_little_endian():
    # matches faiss io.cpp: x[0] | x[1]<<8 | x[2]<<16 | x[3]<<24
    assert fold_rhs('fourcc("rrot")') == ("fourcc", "0x746f7272", 'fourcc("rrot")')
    assert fold_rhs('fourcc("PCAm")') == ("fourcc", "0x6d414350", 'fourcc("PCAm")')
    assert fold_rhs('fourcc("Pcam")') == ("fourcc", "0x6d616350", 'fourcc("Pcam")')


def test_fold_integer_literal():
    assert fold_rhs("12") == ("int", "12", "12")
    assert fold_rhs("0") == ("int", "0", "0")


def test_fold_opaque_callee_keeps_source_not_folded():
    # unknown constant factory → keep source form, no folded concrete value.
    assert fold_rhs('magic("xyz")') == ("opaque_callee", None, 'magic("xyz")')


def test_fold_rejects_non_constant():
    assert fold_rhs("buf") is None
    assert fold_rhs("") is None
    # 4-char factory with a wrong-length literal: not foldable, kept as source (opaque).
    assert fold_rhs('fourcc("toolong")') == ("opaque_callee", None, 'fourcc("toolong")')


# ── parse_equality_leaf ──────────────────────────────────────────────────

def test_parse_single_equality():
    assert parse_equality_leaf('h == fourcc("rrot")') == ("h", "==", 'fourcc("rrot")')
    assert parse_equality_leaf('count != 12') == ("count", "!=", "12")


def test_parse_ignores_disjunctions_and_junk():
    # truncated / multi-term guards are ignored; the individual binary_op leaves carry them.
    assert parse_equality_leaf("h == a || h == b") is None
    assert parse_equality_leaf("x + 1") is None
    assert parse_equality_leaf(None) is None
    assert parse_equality_leaf("") is None


# ── group_domain_facts (grouping + low-noise) ────────────────────────────

def _leaf(var, expr, line=10):
    return ("faiss::read_VectorTransform", line, f"{var} == {expr}")


def test_group_six_fourcc_one_fact():
    leaves = [_leaf("h", f'fourcc("{s}")', 80 + i)
              for i, s in enumerate(["rrot", "PCAm", "LTra", "PcAm", "Viqm", "Pcam"])]
    facts = group_domain_facts(leaves)
    assert len(facts) == 1
    f = facts[0]
    assert f.variable == "h"
    assert f.kind == "fourcc"
    assert len(f.folded) == 6
    assert "0x746f7272" in f.folded


def test_group_single_constant_dropped_as_noise():
    # one comparison does not pin a finite domain.
    assert group_domain_facts([_leaf("h", 'fourcc("rrot")')]) == []


def test_group_opaque_two_distinct_still_emitted():
    leaves = [_leaf("h", 'magic("aa")', 5), _leaf("h", 'magic("bb")', 6)]
    facts = group_domain_facts(leaves)
    assert len(facts) == 1
    assert facts[0].kind == "opaque_callee"
    assert len(facts[0].values) == 2


def test_group_int_variable():
    leaves = [_leaf("mode", "0", 5), _leaf("mode", "1", 6)]
    facts = group_domain_facts(leaves)
    assert len(facts) == 1
    assert facts[0].kind == "int"
    assert sorted(facts[0].folded) == ["0", "1"]


# ── format_domain_facts ──────────────────────────────────────────────────

def test_format_is_advisory_and_contains_hex():
    leaves = [_leaf("h", 'fourcc("rrot")', 82), _leaf("h", 'fourcc("Pcam")', 89)]
    text = format_domain_facts(group_domain_facts(leaves))
    assert "Guard-pinned" in text
    assert "h" in text
    assert "0x746f7272" in text
    # Phase-1: advisory, must NOT hard-assert FP.
    assert "false positive" not in text.lower()
    assert format_domain_facts([]) == ""


# ── end-to-end over real pdg_faiss.json ──────────────────────────────────

def test_real_read_vectortransform_detects_h_domain():
    if not PDG_FAISS.exists():
        return  # faiss PDG not built in this checkout; skip silently.
    pdg = json.loads(PDG_FAISS.read_text())
    fn = next(
        (g for g in pdg["functions"]
         if g.get("function_name") == "faiss::read_VectorTransform"),
        None,
    )
    assert fn is not None, "read_VectorTransform PDG missing"
    leaves = [
        (fn["function_name"], n.get("source_line", 0), n.get("expression", ""))
        for n in fn["nodes"]
        if n.get("expression", "")
    ]
    facts = group_domain_facts(leaves)
    h = next((f for f in facts if f.variable == "h"), None)
    assert h is not None, "expected a domain fact on the header discriminator h"
    # all 10 recognized type-tags (6 in the first guard block + RmDT/VNrm/VCnt/Viqt).
    assert len(h.folded) >= 6
    assert "0x746f7272" in h.folded  # fourcc("rrot")


def test_extract_domain_facts_api_returns_list():
    # API smoke: must accept (sdg, slice_mask, target_fns) and return a list even with None.
    assert extract_domain_facts(None, None, set()) == []


# ── source-text scanning (no SDG dependency, used by Step 3.5 pass) ────────

def test_is_macro_noise_filters_inlined_read_macros():
    assert _is_macro_noise('READ1(pca->x){ size_t ret = (*f)(...)') is True
    assert _is_macro_noise('  if (h == fourcc("rrot")) {') is False
    assert _is_macro_noise('') is False


def test_iter_foldable_equalities_single_and_disjunction():
    assert list(iter_foldable_equalities('if (h == fourcc("rrot")) {')) == \
        [("h", "==", 'fourcc("rrot")')]
    got = list(iter_foldable_equalities(
        'h == fourcc("PCAm") || h == fourcc("PcAm") ||'))
    assert ("h", "==", 'fourcc("PCAm")') in got
    assert ("h", "==", 'fourcc("PcAm")') in got
    # Trailing '))' from a closed guard is not absorbed into the RHS.
    assert list(iter_foldable_equalities('h == fourcc("Pcam")) {')) == \
        [("h", "==", 'fourcc("Pcam")')]


def test_iter_foldable_equalities_handles_inequality_and_int():
    assert list(iter_foldable_equalities('if (h != fourcc("PCAm")) {')) == \
        [("h", "!=", 'fourcc("PCAm")')]
    assert list(iter_foldable_equalities('if (count == 12) {')) == \
        [("count", "==", "12")]


def test_iter_foldable_equalities_ignores_runtime_rhs_and_junk():
    # Comparison against a runtime value (a bare identifier) is not a constant pin.
    assert list(iter_foldable_equalities('if (p == obj) {')) == []
    assert list(iter_foldable_equalities('')) == []
    assert list(iter_foldable_equalities(None)) == []


def test_domain_facts_from_lines_single_var_dropped_as_noise():
    sn = {10: 'if (h == fourcc("rrot")) {'}
    assert domain_facts_from_lines(sn, function_name="f") == []


def test_domain_facts_from_lines_two_guards_pin_h():
    sn = {
        82: 'if (h == fourcc("rrot") || h == fourcc("PCAm") ||',
        83: 'h == fourcc("Pcam")) {',
        85: 'if (h == fourcc("rrot")) {',
    }
    facts = domain_facts_from_lines(sn, function_name="f")
    assert len(facts) == 1
    assert facts[0].variable == "h"
    assert set(facts[0].folded) == {
        "0x746f7272", "0x6d414350", "0x6d616350",  # rrot, PCAm, Pcam
    }


def test_domain_facts_from_lines_skips_macro_noise():
    sn = {
        85: 'if (h == fourcc("rrot")) {',
        91: 'READ1(pca->eigen_power){ size_t ret = (*f)(&(pca->eigen_power), ...',
    }
    facts = domain_facts_from_lines(sn, function_name="f")
    # Only the real control line contributes; macro line is dropped, and a single
    # distinct constant is below the low-noise threshold anyway.
    assert facts == []


def test_domain_facts_from_lines_rejects_runtime_rhs():
    sn = {10: "if (a == b) {", 11: "if (a == c) {"}
    assert domain_facts_from_lines(sn, function_name="f") == []


def test_real_report_snippets_recover_h_domain():
    # Parse the actual read_VectorTransform FP report; its source_snippets must
    # yield `h` pinned to the 6 tags of the outer region (advisory for Step 3.5).
    rpt = REPO_ROOT / ".." / "csa_reports" / "project" / "faiss" / "reports" \
        / "FP" / "report-index_read.cpp-read_VectorTransform-35-1.html"
    if not rpt.exists():
        return  # report checkout not present here; skip silently.
    from llm_client.fp_analysis.html_parser import FPReportParser
    pr = FPReportParser().parse(str(rpt))
    facts = domain_facts_from_lines(pr.source_snippets, function_name="f")
    h = next((f for f in facts if f.variable == "h"), None)
    assert h is not None
    assert len(h.folded) >= 6
    assert "0x746f7272" in h.folded
