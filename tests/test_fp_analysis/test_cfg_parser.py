"""Tests for CFG parser text parsing."""

from __future__ import annotations

from fp_analysis.cfg_models import CFGBlock
from fp_analysis.cfg_parser import parse_cfg_text

# ---------------------------------------------------------------------------
# Fixtures: hardcoded CFG dump text
# ---------------------------------------------------------------------------

CFG_LINEAR = """\
int no_branches()
 [B2 (ENTRY)]
   Succs (1): B1

 [B1]
   1: 42
   2: return [B1.1];
   Preds (1): B2
   Succs (1): B0

 [B0 (EXIT)]
   Preds (1): B1
"""

CFG_IF_ELSE = """\
int test(int x)
 [B4 (ENTRY)]
   Succs (1): B3

 [B1]
   1: 0
   2: return [B1.1];
   Preds (1): B3
   Succs (1): B0

 [B2]
   1: 1
   2: return [B2.1];
   Preds (1): B3
   Succs (1): B0

 [B3]
   1: x
   2: [B3.1] (ImplicitCastExpr, LValueToRValue, int)
   3: 0
   4: [B3.2] > [B3.3]
   T: if [B3.4]
   Preds (1): B4
   Succs (2): B2 B1

 [B0 (EXIT)]
   Preds (2): B1 B2
"""

CFG_NESTED_IF = """\
int nested(int a, int b)
 [B5 (ENTRY)]
   Succs (1): B4

 [B1]
   1: 1
   2: return [B1.1];
   Preds (1): B3
   Succs (1): B0

 [B2]
   1: 2
   2: return [B2.1];
   Preds (1): B4
   Succs (1): B0

 [B3]
   1: b
   2: [B3.1] (ImplicitCastExpr, LValueToRValue, int)
   3: 0
   4: [B3.2] > [B3.3]
   T: if [B3.4]
   Preds (1): B4
   Succs (2): B1 B4

 [B4]
   1: a
   2: [B4.1] (ImplicitCastExpr, LValueToRValue, int)
   3: 0
   4: [B4.2] > [B4.3]
   T: if [B4.4]
   Preds (2): B5 B3
   Succs (2): B3 B2

 [B0 (EXIT)]
   Preds (2): B1 B2
"""

CFG_WITH_LOOP = """\
int loop(int n)
 [B5 (ENTRY)]
   Succs (1): B4

 [B1]
   1: i
   2: [B1.1] (ImplicitCastExpr, LValueToRValue, int)
   3: n
   4: [B1.3] (ImplicitCastExpr, LValueToRValue, int)
   5: [B1.2] < [B1.4]
   T: [B1.5] (ImplicitCastExpr, IntegralToBoolean, int)
   Preds (2): B4 B3
   Succs (2): B2 B4

 [B2]
   1: i
   2: [B2.1] (ImplicitCastExpr, LValueToRValue, int)
   3: ++ [B2.2]
   Preds (1): B1
   Succs (1): B3

 [B3]
   1: (B3)
   Preds (1): B2
   Succs (1): B1

 [B4]
   1: 0
   2: int i = 0;
   3: i
   4: [B4.2] (ImplicitCastExpr, LValueToRValue, int)
   T: [B4.2] = [B4.1]
   Preds (1): B5
   Succs (1): B1

 [B0 (EXIT)]
   Preds (1): B4
"""

# ---------------------------------------------------------------------------
# Tests: parse_cfg_text
# ---------------------------------------------------------------------------


class TestParseCfgText:
    def test_linear_function(self):
        """Parse a function with no branches."""
        cfgs = parse_cfg_text(CFG_LINEAR)
        assert "int no_branches()" in cfgs
        cfg = cfgs["int no_branches()"]
        assert len(cfg.blocks) == 3  # B2 (entry), B1, B0 (exit)
        assert "B2" in cfg.blocks
        assert "B1" in cfg.blocks
        assert "B0" in cfg.blocks
        assert cfg.entry_block_id == "B1"  # B2's successor
        assert cfg.exit_block_id == "B0"

    def test_if_else_function(self):
        """Parse a function with an if-else branch."""
        cfgs = parse_cfg_text(CFG_IF_ELSE)
        assert "int test(int x)" in cfgs
        cfg = cfgs["int test(int x)"]
        assert len(cfg.blocks) == 5
        # B3 should have the terminator and 2 successors
        b3 = cfg.blocks["B3"]
        assert b3.terminator is not None
        assert "if" in b3.terminator
        assert b3.successors == ["B2", "B1"]
        assert len(b3.statements) == 4

    def test_nested_if_function(self):
        """Parse a function with nested if-else."""
        cfgs = parse_cfg_text(CFG_NESTED_IF)
        assert "int nested(int a, int b)" in cfgs
        cfg = cfgs["int nested(int a, int b)"]
        assert len(cfg.blocks) == 6
        # B4 is the outer if, B3 is the inner if
        assert cfg.blocks["B4"].successors == ["B3", "B2"]
        assert cfg.blocks["B3"].successors == ["B1", "B4"]

    def test_loop_function(self):
        """Parse a function with a loop."""
        cfgs = parse_cfg_text(CFG_WITH_LOOP)
        assert "int loop(int n)" in cfgs
        cfg = cfgs["int loop(int n)"]
        # Loops have back edges: B1 loops back to B4 via B3/B2
        b1 = cfg.blocks["B1"]
        assert b1.terminator is not None  # loop condition
        assert len(b1.successors) == 2  # B2 (body) and B4 (exit)

    def test_empty_input(self):
        """Empty input produces empty dict."""
        assert parse_cfg_text("") == {}
        assert parse_cfg_text("   \n  \n") == {}

    def test_no_blocks(self):
        """Text with no [B blocks produces empty CFG."""
        cfgs = parse_cfg_text("just some random text\nwith no cfg blocks")
        assert cfgs == {}

    def test_entry_without_explicit_label(self):
        """Blocks without explicit ENTRY/EXIT labels are handled."""
        # Some clang versions don't label ENTRY/EXIT
        text = CFG_LINEAR.replace("(ENTRY)", "").replace("(EXIT)", "")
        cfgs = parse_cfg_text(text)
        assert cfgs  # should still parse
        cfg = list(cfgs.values())[0]
        assert cfg.entry_block_id  # should have a best-guess entry
        assert cfg.exit_block_id == "B0"


class TestCFGBlock:
    """Sanity checks on CFGBlock model."""

    def test_minimal_block(self):
        block = CFGBlock(block_id="B1")
        assert block.block_id == "B1"
        assert block.label == ""
        assert block.statements == []
        assert block.terminator is None
        assert block.predecessors == []
        assert block.successors == []

    def test_full_block(self):
        block = CFGBlock(
            block_id="B3",
            label="ENTRY",
            statements=["x", "0", "x > 0"],
            terminator="if [B3.3]",
            predecessors=["B4"],
            successors=["B2", "B1"],
        )
        assert block.block_id == "B3"
        assert block.label == "ENTRY"
        assert len(block.statements) == 3
        assert block.terminator == "if [B3.3]"
        assert block.predecessors == ["B4"]
        assert block.successors == ["B2", "B1"]
