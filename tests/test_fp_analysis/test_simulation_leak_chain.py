"""Deterministic mapping tests for the memory-leak `leak_chain` verdict layer.

These never call an LLM: they drive ``_leak_decisive_conclusion`` / ``_compose_leak_reason`` and the
leak-type detector with hand-built :class:`SimulationResult` objects, so they guard the leak FP/TP
policy (see the MEMORY-LEAK SIMULATION rules in prompt_templates.py) against regression.
"""

from llm_client.fp_analysis.simulation_verifier import (
    SimulationResult,
    SimulationVerifier,
    _is_leak_type,
    _normalize_leak_chain,
)


def _verdict(leak_chain):
    sim = SimulationResult(
        triggered=False, reason="sim said ...", leak_chain=_normalize_leak_chain(leak_chain)
    )
    # _leak_decisive_conclusion / _compose_leak_reason do not touch self._analyzer.
    verifier = SimulationVerifier(None)
    return verifier._leak_decisive_conclusion(sim)


def test_leak_type_detector():
    assert _is_leak_type("Memory leak")
    assert _is_leak_type("Potential leak of memory")
    assert _is_leak_type("Memory Leak")  # casing-insensitive
    assert not _is_leak_type("Dereference of null pointer")
    assert not _is_leak_type("")
    assert not _is_leak_type(None)


def test_leak_fp_stored_to_return_broken_input_only():
    # 484-2 shape: object stored into the return on every well-formed continuation; the only
    # abandon is a mid-stream throw needing corrupt/truncated input → FP.
    assert _verdict({
        "wellformed_continuation_stores_to": "return",
        "requires_broken_input": True,
        "leak_fires_on_wellformed": False,
        "abandon_path": "exception_after_alloc",
    }) == "fp"


def test_leak_fp_freed_on_any_path():
    assert _verdict({"freed_on_any_path": True}) == "fp"


def test_leak_fp_no_abandon_always_owned():
    assert _verdict({"wellformed_continuation_stores_to": "owner", "abandon_path": None}) == "fp"


def test_leak_tp_genuine_wellformed_abandon():
    assert _verdict({"leak_fires_on_wellformed": True}) == "tp"


def test_leak_undecided_cases_return_none():
    assert _verdict(None) is None
    # abandoned, but NOT via broken input and NOT stored → genuinely ambiguous, undecided.
    assert _verdict({
        "abandon_path": "manual_no_free_early_return",
        "wellformed_continuation_stores_to": None,
        "requires_broken_input": False,
    }) is None


def test_compose_leak_reason_is_grounded():
    verifier = SimulationVerifier(None)
    lc = _normalize_leak_chain({
        "alloc_expr": "new IndexNSGPQ()",
        "wellformed_continuation_stores_to": "return",
        "requires_broken_input": True,
    })
    reason = verifier._compose_leak_reason(lc, SimulationResult(triggered=False, reason="x"))
    assert "FP" in reason and "new IndexNSGPQ()" in reason
