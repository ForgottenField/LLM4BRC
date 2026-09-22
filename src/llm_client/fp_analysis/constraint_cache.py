"""Disk-backed cache for mined project-wide constraints.

Mirrors ``SummaryCache`` pattern::

    constraint_cache/
    └── <project_name>.json   # ProjectConstraintDB
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from llm_client.fp_analysis.constraint_models import ProjectConstraintDB

logger = logging.getLogger(__name__)


class ConstraintCache:
    """Persistent JSON cache for mined project-wide constraints."""

    def __init__(self, cache_root: str | Path = "constraint_cache") -> None:
        self._cache_root = Path(cache_root).resolve()

    def get(self, project_name: str) -> ProjectConstraintDB | None:
        """Load cached constraint DB for *project_name*."""
        cache_path = self._cache_path(project_name)
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return ProjectConstraintDB(**data)
        except Exception as e:
            logger.warning("Failed to load constraint cache %s: %s", cache_path, e)
            return None

    def set(self, project_name: str, db: ProjectConstraintDB) -> None:
        """Save constraint DB to disk atomically."""
        cache_path = self._cache_path(project_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json", dir=cache_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(db.model_dump_json(indent=2))
            os.replace(tmp_path, cache_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        logger.info("Constraint cache saved: %s", cache_path)

    def has(self, project_name: str) -> bool:
        return self._cache_path(project_name).exists()

    def clear(self, project_name: str | None = None) -> None:
        if project_name:
            self._cache_path(project_name).unlink(missing_ok=True)
        elif self._cache_root.exists():
            for f in self._cache_root.iterdir():
                f.unlink()

    def _cache_path(self, project_name: str) -> Path:
        return self._cache_root / f"{project_name}.json"
