#!/usr/bin/env python3
"""Classify a Clang Static Analyzer (CSA) bug report as TP or FP via DeepSeek.

A minimal, dependency-free classifier:
  * Reads DEEPSEEK_API_KEY from ./.env (no python-dotenv needed).
  * Extracts the bug-path narrative out of the scan-build HTML report using
    only the Python standard library (html.parser).
  * Calls the DeepSeek chat-completion HTTP API directly with urllib.

Runtime needed: Python 3.10 (stdlib only) + a valid DEEPSEEK_API_KEY.
No pip packages are required.

Usage:
    python3.10 deepseek_tp_fp.py --report <path/to/report.html>
    python3.10 deepseek_tp_fp.py --report <path> --model deepseek-chat --max-chars 8000
"""

from __future__ import annotations

import argparse
import html.parser
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE / ".env"

# DeepSeek is OpenAI-compatible; default model matches the FP pipeline.
DEFAULT_MODEL = "deepseek-chat"
API_URL = "https://api.deepseek.com/chat/completions"


# ───────────────────────────────────────────────────────────────────────
# Report text extraction (stdlib only)
# ───────────────────────────────────────────────────────────────────────


class ReportExtractor(html.parser.HTMLParser):
    """Pull the bug-path narrative out of a scan-build HTML report.

    Captures:
      * the ``Bug Summary`` heading (bug type + event summary)
      * every path event (``class="msg"`` blocks), each carrying its
        ordered ``PathIndex`` and a human-readable message.
    """

    def __init__(self) -> None:
        super().__init__()
        self._skip = 0  # depth inside <style>/<script> (their text is noise)
        self._in_msg = False
        self._buf: list[str] = []
        self.events: list[str] = []
        self.headings: list[str] = []
        self._heading_tag: str | None = None
        self._heading_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("style", "script"):
            self._skip += 1
        if self._skip:
            return
        cls = dict(attrs).get("class", "") or ""
        if "msg" in cls.split():
            self._in_msg = True
        elif tag in ("h1", "h2", "h3", "h4"):
            self._heading_tag = tag
            self._heading_buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("style", "script"):
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag == "table" and self._in_msg:
            # Each msg block wraps exactly one <table>; flush the event text.
            text = " ".join(" ".join(self._buf).split())
            if text:
                self.events.append(text)
            self._buf = []
            self._in_msg = False
        elif tag in ("h1", "h2", "h3", "h4") and tag == self._heading_tag:
            text = " ".join(" ".join(self._heading_buf).split())
            if text:
                self.headings.append(text)
            self._heading_tag = None

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_msg:
            self._buf.append(data)
        if self._heading_tag:
            self._heading_buf.append(data)

    def narrative(self, max_chars: int) -> str:
        parts = ["### Bug summary", *self.headings, "\n### Path events"]
        parts += [f"{i + 1}. {e}" for i, e in enumerate(self.events)]
        text = "\n".join(parts)
        return text[:max_chars]


def extract_report_text(report_path: str, max_chars: int) -> str:
    parser = ReportExtractor()
    parser.feed(Path(report_path).read_text(encoding="utf-8", errors="replace"))
    return parser.narrative(max_chars)


# ───────────────────────────────────────────────────────────────────────
# DeepSeek API call (stdlib urllib)
# ───────────────────────────────────────────────────────────────────────


def load_api_key() -> str:
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    key = __import__("os").environ.get("DEEPSEEK_API_KEY", "")
    return key


def build_messages(report_text: str) -> list[dict]:
    system = (
        "You are an expert in Clang Static Analyzer bug reports. Given a CSA "
        "report's bug-path narrative, decide whether the reported bug is a "
        "TRUE POSITIVE (TP, the bug path is genuinely reachable) or a FALSE "
        "POSITIVE (FP, the path is not actually reachable / a false alarm). "
        "Reply with exactly one line starting with TP or FP, then a brief "
        "reason. Example: 'FP: the error-start branch is guarded by an "
        "infeasible condition (x <= 0).'"
    )
    user = (
        "Classify the following CSA bug report as TP or FP:\n\n"
        f"---\n{report_text}\n---"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def classify(report_text: str, api_key: str, model: str, timeout: float) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": build_messages(report_text),
            "temperature": 0.0,
            "max_tokens": 256,
            "stream": False,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    try:
        return payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected DeepSeek response: {payload}") from exc


# ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify a CSA report as TP/FP via DeepSeek.")
    parser.add_argument("--report", required=True, help="Path to the CSA HTML report")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model id")
    parser.add_argument("--max-chars", type=int, default=8000, help="Max report chars to send")
    parser.add_argument("--timeout", type=float, default=120.0, help="API timeout seconds")
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        print("✗ DEEPSEEK_API_KEY not found in .env or environment", file=sys.stderr)
        return 1

    report_text = extract_report_text(args.report, args.max_chars)
    print(f"model     : {args.model}")
    print(f"report    : {args.report}")
    print(f"events    : {args.max_chars} chars -> sent {len(report_text)} chars")
    print("─" * 60)
    print("→ calling DeepSeek ...")

    try:
        verdict = classify(report_text, api_key, args.model, args.timeout)
    except Exception as exc:  # network / API errors surface here
        print(f"✗ API call failed: {exc}", file=sys.stderr)
        return 1

    print("─" * 60)
    print("VERDICT")
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
