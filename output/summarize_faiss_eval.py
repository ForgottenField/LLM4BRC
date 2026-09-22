#!/usr/bin/env python3
"""Summarize a faiss pipeline run: classification accuracy + path-space details.

Usage:
    python3.10 output/summarize_faiss_eval.py [results_dir]

Reads ``output/path_selection_report-*.json`` (written by the default flow, which
always runs POC verification) and prints one row per report:

    ground truth (TP/FP from the report's parent directory) | verdict | judgement
    segments / branches / relevant / whole-path bound / feasible candidates / attempts

Ground truth comes from the corpus layout: ``reports/TP/`` are true positives,
``reports/FP/`` are false positives.  A verdict of UNKNOWN is neither.

The path-space columns are *deterministic* (computed before the verification
stage), so unlike the verdict they do not depend on LLM rounds.  Two of them are
routinely misread, and the script annotates both:

- ``bound==1`` — the whole path space collapsed to a single equivalence class.
  Legitimate for a genuinely branch-free path, but also the signature of the
  event→file attribution collapse and of an over-eager compression.
- ``capped`` — the reported bound is a *lower* bound: some segment hit
  ``ShortestPathComputer._MAX_CANDIDATES`` (200), so the product is truncated.
  The pipeline does not enumerate a capped space; the number is not a measurement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "output"
CORPUS = Path.home() / "csa_reports" / "project" / "faiss" / "reports"

# Ground truth per report name, from the corpus layout.
GROUND_TRUTH: dict[str, str] = {}
for split in ("TP", "FP"):
    for html in (CORPUS / split).glob("report-*.html"):
        GROUND_TRUTH[html.name] = split


def _ps(d: dict) -> dict:
    return d.get("path_space") or {}


def _rel(d: dict) -> dict:
    return ((d.get("slice_relevance") or {}).get("branch_stats") or {})


def collect(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("path_selection_report-*.json")):
        d = json.loads(path.read_text())
        report = d.get("report") or path.name[len("path_selection_"):-len(".json")]
        ps = _ps(d)
        bf = d.get("branch_filter") or {}
        attempts = ps.get("attempts_used")
        max_attempts = ps.get("max_path_attempts")
        rows.append({
            "report": report,
            "truth": GROUND_TRUTH.get(report, "?"),
            "verdict": d.get("verdict"),
            "step": d.get("verdict_step") or "",
            "segments": bf.get("num_segments"),
            "branches": bf.get("total_branches"),
            "relevant": _rel(d).get("num_relevant"),
            "bound": ps.get("whole_path_bound"),
            "capped": ps.get("capped"),
            "candidates": ps.get("total_feasible_candidates"),
            "attempts": attempts,
            "max_attempts": max_attempts,
            "vstatus": d.get("verification_status"),
            "flipped": d.get("verification_flipped_to_fp"),
        })
    return rows


def judgement(row: dict) -> str:
    truth, verdict = row["truth"], row["verdict"]
    if truth == "?":
        return "?"
    if verdict == "UNKNOWN":
        return "UNDECIDED"
    return "CORRECT" if truth == verdict else "WRONG"


def flags(row: dict) -> str:
    out = []
    if row["bound"] == 1:
        out.append("bound==1")
    if row["capped"]:
        out.append("capped")
    if row["branches"] == 0:
        out.append("0-branch")
    # ``max_path_attempts`` is the EFFECTIVE cap for this run (often 1, i.e. the
    # loop finished inside its budget) — ``attempts >= cap`` therefore does not
    # mean exhaustion.  Hitting the traversal limit surfaces as an UNKNOWN
    # verdict whose step says so.
    if "遍历超限" in row["step"]:
        out.append("traversal-limit")
    return ",".join(out)


def main(argv: list[str]) -> int:
    results_dir = Path(argv[1]) if len(argv) > 1 else OUT
    rows = collect(results_dir)
    if not rows:
        print(f"no path_selection_report-*.json in {results_dir}")
        return 1

    head = (f"{'report':46s} {'truth':>5s} {'verdict':>8s} {'judge':>10s} "
            f"{'seg':>4s} {'br':>5s} {'rel':>4s} {'bound':>12s} {'cand':>6s} "
            f"{'att':>5s}  flags")
    print(head)
    print("-" * len(head))
    for row in rows:
        bound = row["bound"]
        print(f"{row['report']:46s} {row['truth']:>5s} {str(row['verdict']):>8s} "
              f"{judgement(row):>10s} {str(row['segments']):>4s} "
              f"{str(row['branches']):>5s} {str(row['relevant']):>4s} "
              f"{(f'{bound:,}' if isinstance(bound, int) else str(bound)):>12s} "
              f"{str(row['candidates']):>6s} {str(row['attempts']):>5s}  {flags(row)}")

    # ── Accuracy ──
    # Only rows whose ground truth is known (i.e. present in the corpus tree's
    # TP/FP dirs) take part in the accuracy tally; results_dir may also hold
    # reports from other projects.
    scored = [r for r in rows if r["truth"] in ("TP", "FP")]
    print("\n=== classification ===")
    n_tp = sum(1 for r in scored if r["truth"] == "TP")
    n_fp = sum(1 for r in scored if r["truth"] == "FP")
    print(f"corpus: {len(scored)} reports ({n_tp} TP / {n_fp} FP)"
          + (f" | {len(rows) - len(scored)} rows with unknown ground truth ignored"
             if len(rows) != len(scored) else ""))
    for truth in ("TP", "FP"):
        sub = [r for r in scored if r["truth"] == truth]
        if not sub:
            continue
        correct = sum(1 for r in sub if judgement(r) == "CORRECT")
        undecided = sum(1 for r in sub if judgement(r) == "UNDECIDED")
        wrong = len(sub) - correct - undecided
        detail = ", ".join(f"{r['report'][7:-5]}→{r['verdict']}"
                           for r in sub if judgement(r) != "CORRECT")
        print(f"  {truth}: correct {correct}/{len(sub)} | undecided {undecided} "
              f"| wrong {wrong}")
        if detail:
            print(f"     not correct: {detail}")

    print("\n=== path-space caveats ===")
    for row in rows:
        notes = []
        if row["bound"] == 1:
            notes.append("bound==1 — collapsed or genuinely branch-free (check "
                         "segments/branches)")
        if row["capped"]:
            notes.append(f"capped — a segment hit the 200-candidate cap, so the "
                         f"bound is a truncated LOWER bound")
        if "遍历超限" in row["step"]:
            notes.append(f"traversal limit reached after {row['attempts']} "
                         f"attempt(s) — UNKNOWN is the honest verdict here")
        if notes:
            print(f"  {row['report']}")
            for n in notes:
                print(f"     · {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
