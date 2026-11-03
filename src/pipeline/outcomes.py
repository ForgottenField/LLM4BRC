"""Typed outcome vocabulary for the application layer.

These values describe the public result contract.  They deliberately do not
implement verdict policy: stages still produce the same evidence and the same
control flow, while the names below replace untyped string literals at module
boundaries.
"""

from __future__ import annotations

from enum import Enum


class Verdict(str, Enum):
    """Terminal states emitted by one report run."""

    TP = "TP"
    FP = "FP"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"

    @classmethod
    def coerce(cls, value: "Verdict | str | None") -> "Verdict | None":
        if value is None or isinstance(value, cls):
            return value
        return cls(str(value).upper())


class VerificationConclusion(str, Enum):
    """Internal conclusion from the stage-10 verification loop."""

    TP = "tp"
    FP = "fp"
    UNRESOLVED = "unresolved"
