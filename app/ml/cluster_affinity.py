"""Cluster → book affinity scores for hybrid recommendations."""

from __future__ import annotations

import json
from pathlib import Path

from app.logging_config import get_logger

logger = get_logger(__name__)


class ClusterAffinityStore:
    """Maps cluster_id to normalized book affinity scores (source_book_id → 0..1)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._by_cluster: dict[int, dict[str, float]] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        if self.path is None or not self.path.is_file():
            logger.warning("Cluster affinity file not found: %s", self.path)
            self._loaded = True
            return False
        with self.path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        self._by_cluster = {
            int(cluster_id): {str(book_id): float(score) for book_id, score in books.items()}
            for cluster_id, books in raw.items()
        }
        self._loaded = True
        logger.info(
            "Loaded cluster affinity for %s clusters from %s",
            len(self._by_cluster),
            self.path,
        )
        return bool(self._by_cluster)

    def score(self, cluster_id: int, source_book_id: str) -> float:
        return self._by_cluster.get(cluster_id, {}).get(str(source_book_id), 0.0)

    def top_books(self, cluster_id: int, *, limit: int = 80) -> list[tuple[str, float]]:
        items = list(self._by_cluster.get(cluster_id, {}).items())
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:limit]

    def all_clusters(self) -> list[int]:
        return sorted(self._by_cluster.keys())
