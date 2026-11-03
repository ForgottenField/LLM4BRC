"""API credential loading from the repository's JSON configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_client.config import provider_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_KEY_FILE = REPO_ROOT / "config" / "api_key.json"


class CredentialConfigError(ValueError):
    """The credential file is missing, invalid, or lacks the requested key."""


def load_api_key(
    provider_name: str,
    api_key_file: str | Path | None = None,
) -> str:
    """Load one provider API key without exposing its value in errors."""
    path = Path(api_key_file).expanduser() if api_key_file else DEFAULT_API_KEY_FILE
    if not path.is_file():
        raise CredentialConfigError(f"API key file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CredentialConfigError(f"cannot read API key file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CredentialConfigError(f"API key file must contain a JSON object: {path}")

    spec = provider_spec(provider_name)
    candidates: tuple[str, ...] = (
        spec.api_key_env,
        provider_name,
        provider_name.lower(),
        f"{provider_name.lower()}_api_key",
        f"{provider_name.upper()}_API_KEY",
    )
    for name in dict.fromkeys(candidates):
        value = _key_value(data.get(name))
        if value:
            return value

    raise CredentialConfigError(
        f"API key file {path} has no non-empty key for provider "
        f"{provider_name!r}; expected one of {', '.join(dict.fromkeys(candidates))}"
    )


def _key_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("api_key", "key", "token"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""
