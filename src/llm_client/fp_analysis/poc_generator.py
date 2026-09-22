"""POC generation from completed execution paths."""

from __future__ import annotations

import logging
from pathlib import Path

from llm_client.fp_analysis.models import CompletedPath

logger = logging.getLogger(__name__)


class POCGenerator:
    """Validate and format POC test harnesses generated from completed paths."""

    def validate_syntax(self, code: str) -> list[str]:
        """Basic syntax validation of generated C++ code.

        Checks for obvious issues like:
        - Mismatched braces
        - Missing includes
        - Common errors

        Returns a list of warnings (empty = no issues found).
        """
        warnings: list[str] = []

        lines = code.split("\n")
        # Check brace balance
        open_braces = 0
        for i, line in enumerate(lines):
            open_braces += line.count("{")
            open_braces -= line.count("}")
            if open_braces < 0:
                warnings.append(f"Line {i+1}: unexpected closing brace")
                break

        if open_braces > 0:
            warnings.append(f"Unclosed braces ({open_braces} unclosed)")

        # Check for common issues
        if "#include" not in code:
            warnings.append("No #include directives found")

        if "main" not in code and "TEST(" not in code and "TEST_F(" not in code:
            warnings.append("No test entry point (main/TEST) found")

        if "folly/" in code and "#include <folly/" not in code:
            warnings.append("References folly/ headers but no proper include found")

        return warnings

    def format_code(self, code: str) -> str:
        """Format/clean up generated POC code."""
        # Remove leading/trailing whitespace
        code = code.strip()

        # Ensure there's a trailing newline
        if not code.endswith("\n"):
            code += "\n"

        return code

    def save_poc(self, code: str, output_path: str | Path) -> Path:
        """Save POC code to a file.

        Args:
            code: C++ source code.
            output_path: Path to write the file.

        Returns:
            The output path.
        """
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        logger.info("POC saved to %s", path)
        return path
