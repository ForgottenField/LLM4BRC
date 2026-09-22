#!/usr/bin/env python3
"""Report macro-expanded branch statistics from a PDG JSON built by PDGBuilder.

The C++ PDG builder marks every node whose text came from a macro expansion
(``is_macro`` + ``macro_name`` / ``macro_def_file`` / ``macro_def_line``) and
every control-dependence predicate that is macro-derived
(``is_macro_branch`` on the edge, ``macro_branches`` per function).  This
script turns those marks into the numbers we care about:

  * how many branches the project has in total,
  * how many of them exist only because a macro was expanded,
  * which macros contribute the most branches,
  * which macro definition sites they come from.

Usage:
    python3 macro_branch_stats.py                          # pdg_faiss.json
    python3 macro_branch_stats.py --pdg pdg_folly.json
    python3 macro_branch_stats.py --top 20
    python3 macro_branch_stats.py --function read_VectorTransform
    python3 macro_branch_stats.py --json                   # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter


def _base_name(qualified: str) -> str:
    """``faiss::read_index`` -> ``read_index`` (drops namespace, no params)."""
    name = qualified.split("(")[0].strip()
    return name.split("::")[-1] if name else ""


def load(path: str) -> dict:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


def compute_stats(pdg: dict) -> dict:
    """Aggregate macro statistics.

    Prefers the ``macro_stats`` block the C++ builder emits; recomputes from
    the per-function data when it is absent (older PDG files), so this script
    also works on PDGs built before the annotation existed — it then simply
    reports zeros, which is the honest answer for those files.
    """
    if "macro_stats" in pdg:
        return pdg["macro_stats"]

    total_nodes = macro_nodes = 0
    total_branches = macro_branches = 0
    by_name: Counter[str] = Counter()
    by_site: Counter[str] = Counter()
    funcs_with_macro_branches = funcs_with_macro_nodes = 0

    for fn in pdg.get("functions", []):
        f_macro_nodes = 0
        for n in fn.get("nodes", []):
            total_nodes += 1
            if n.get("is_macro"):
                macro_nodes += 1
                f_macro_nodes += 1
        if f_macro_nodes:
            funcs_with_macro_nodes += 1

        sources = {e.get("source_id") for e in fn.get("control_dep_edges", [])}
        total_branches += len(sources)

        per_fn_branches = fn.get("macro_branches", [])
        macro_branches += len(per_fn_branches)
        if per_fn_branches:
            funcs_with_macro_branches += 1
        for b in per_fn_branches:
            by_name[b.get("macro_name") or "<unnamed>"] += 1
            site = b.get("macro_def_file") or "<unknown>"
            by_site[f"{site}:{b.get('macro_def_line', 0)}"] += 1

    def ratio(a: int, b: int) -> float:
        return round(a / b, 4) if b else 0.0

    return {
        "_recomputed": True,  # no macro_stats block → marks predate annotation
        "function_count": len(pdg.get("functions", [])),
        "total_node_count": total_nodes,
        "macro_node_count": macro_nodes,
        "total_branch_count": total_branches,
        "macro_branch_count": macro_branches,
        "function_count_with_macro_nodes": funcs_with_macro_nodes,
        "function_count_with_macro_branches": funcs_with_macro_branches,
        "macro_node_ratio": ratio(macro_nodes, total_nodes),
        "macro_branch_ratio": ratio(macro_branches, total_branches),
        "macro_branches_by_name": {
            k: v for k, v in by_name.most_common()
        },
        "macro_def_sites": {k: v for k, v in by_site.most_common()},
    }


def _short(path: str, limit: int = 46) -> str:
    """Trim a path from the left so the interesting tail stays visible."""
    return path if len(path) <= limit else "..." + path[-(limit - 3):]


def print_report(stats: dict, pdg: dict, top: int) -> None:
    total_b = stats.get("total_branch_count", 0)
    macro_b = stats.get("macro_branch_count", 0)
    # A file without the builder's macro_stats block predates the annotation:
    # the zeros below mean "not recorded", not "no macros".
    note = ""
    if stats.get("_recomputed"):
        note = ("\n  ⚠ 该 PDG 无 macro_stats 字段(构建于标注功能之前),"
                "0 表示「未记录」而非「无宏」。请用新版 PDGBuilder 重建。")

    print("=" * 72)
    print("宏展开分支统计 (macro-expanded branch statistics)")
    print("=" * 72)
    print(f"  函数数            : {stats.get('function_count', 0)}")
    print(f"  节点数 / 宏生成节点: {stats.get('total_node_count', 0)} / "
          f"{stats.get('macro_node_count', 0)}"
          f"  ({stats.get('macro_node_ratio', 0):.1%})")
    print(f"  分支数 / 宏生成分支: {total_b} / {macro_b}"
          f"  ({stats.get('macro_branch_ratio', 0):.1%}){note}")
    print(f"  含宏节点的函数    : {stats.get('function_count_with_macro_nodes', 0)}")
    print(f"  含宏分支的函数    : {stats.get('function_count_with_macro_branches', 0)}")

    by_name = stats.get("macro_branches_by_name", {})
    if by_name:
        print(f"\n-- 按宏名 (top {top}) " + "-" * 40)
        rows = sorted(
            by_name.items(),
            key=lambda kv: -(kv[1].get("branch_count", 0)
                             if isinstance(kv[1], dict) else kv[1]),
        )[:top]
        for name, val in rows:
            if isinstance(val, dict):
                n_b, n_f = val.get("branch_count", 0), val.get("function_count", 0)
            else:
                n_b, n_f = val, ""
            extra = f"  ({n_f} funcs)" if n_f != "" else ""
            print(f"  {n_b:7d}  {name}{extra}")

    by_site = stats.get("macro_def_sites", {})
    if by_site:
        print(f"\n-- 按宏定义点 (top {top}) " + "-" * 36)
        for site, cnt in sorted(by_site.items(), key=lambda kv: -kv[1])[:top]:
            path, _, line = site.rpartition(":")
            print(f"  {cnt:7d}  {_short(path)}:{line}")


def print_function(pdg: dict, name: str) -> None:
    """Print every macro branch of one function, in call-site order."""
    matches = [f for f in pdg.get("functions", [])
               if name == f.get("function_name") or name == _base_name(
                   f.get("function_name", ""))]
    if not matches:
        print(f"function not found: {name}", file=sys.stderr)
        sys.exit(1)

    for fn in matches:
        branches = sorted(fn.get("macro_branches", []),
                          key=lambda b: b.get("call_site_line", 0))
        print("=" * 72)
        print(f"{fn.get('function_name')}  ({fn.get('source_file')})")
        print("=" * 72)
        st = fn.get("macro_stats", {})
        total_b, macro_b = st.get("total_branch_count", 0), st.get("macro_branch_count", 0)
        print(f"  分支 {total_b} / 其中宏生成 {macro_b}"
              f"  ({st.get('macro_branch_ratio', 0):.1%})")
        if not branches:
            print("  (该函数没有宏生成分支)")
            continue
        print(f"\n  {'调用行':>6}  {'宏名':<16} {'定义点':<28} 展开后的条件")
        for b in branches:
            site = (f"{_short(b.get('macro_def_file', ''), 22)}"
                    f":{b.get('macro_def_line', 0)}")
            expr = b.get("expression", "")
            if len(expr) > 40:
                expr = expr[:37] + "..."
            print(f"  {b.get('call_site_line', 0):>6}  "
                  f"{b.get('macro_name', '') or '<unnamed>':<16} {site:<28} {expr}")


def print_top_functions(pdg: dict, top: int) -> None:
    rows = []
    for fn in pdg.get("functions", []):
        st = fn.get("macro_stats", {})
        if st.get("macro_branch_count"):
            rows.append((st["macro_branch_count"],
                         st.get("total_branch_count", 0),
                         fn.get("function_name", "")))
    if not rows:
        return
    print(f"\n-- 宏分支最多的函数 (top {top}) " + "-" * 34)
    for macro_b, total_b, fn_name in sorted(rows, reverse=True)[:top]:
        print(f"  {macro_b:5d}/{total_b:<5d}  {fn_name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdg", default="pdg_faiss.json",
                    help="PDG JSON produced by PDGBuilder (default: pdg_faiss.json)")
    ap.add_argument("--top", type=int, default=15,
                    help="rows to show in each ranked table (default: 15)")
    ap.add_argument("--function", help="dump every macro branch of one function")
    ap.add_argument("--json", action="store_true",
                    help="print raw statistics as JSON")
    args = ap.parse_args()

    if not os.path.exists(args.pdg):
        print(f"PDG file not found: {args.pdg}", file=sys.stderr)
        return 1

    pdg = load(args.pdg)
    stats = compute_stats(pdg)

    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    if args.function:
        print_function(pdg, args.function)
        return 0

    print_report(stats, pdg, args.top)
    print_top_functions(pdg, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
