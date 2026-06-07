"""Global and per-cluster genre priors for cold-start users."""

from __future__ import annotations

import json
from pathlib import Path

from app.logging_config import get_logger

logger = get_logger(__name__)


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {g: w / total for g, w in weights.items()}


class GenrePriorStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._global: dict[str, float] = {}
        self._clusters: dict[int, dict[str, float]] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        if self.path is None or not self.path.is_file():
            logger.warning("Genre priors file not found: %s", self.path)
            self._loaded = True
            return False
        with self.path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        self._global = _normalize({str(k): float(v) for k, v in (raw.get("global") or {}).items()})
        self._clusters = {
            int(cid): _normalize({str(k): float(v) for k, v in genres.items()})
            for cid, genres in (raw.get("clusters") or {}).items()
        }
        self._loaded = True
        logger.info(
            "Loaded genre priors: global=%s genres, clusters=%s",
            len(self._global),
            len(self._clusters),
        )
        return bool(self._global or self._clusters)

    def for_cluster(self, cluster_id: int, *, global_weight: float = 0.35) -> dict[str, float]:
        """Blend global + cluster genre distribution for users with no ratings."""
        cluster_w = self._clusters.get(cluster_id, {})
        if not self._global and not cluster_w:
            return {}
        blended: dict[str, float] = {}
        keys = set(self._global) | set(cluster_w)
        cw = 1.0 - global_weight
        for genre in keys:
            blended[genre] = global_weight * self._global.get(genre, 0.0) + cw * cluster_w.get(genre, 0.0)
        return _normalize(blended)

    def top_genres(self, cluster_id: int, *, limit: int = 5) -> list[str]:
        weights = self.for_cluster(cluster_id)
        return sorted(weights, key=weights.get, reverse=True)[:limit]
