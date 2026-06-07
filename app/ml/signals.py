"""Signal taxonomy — ML models vs heuristics (for defense / API transparency)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SignalKind(str, Enum):
    ML = "ml"
    HEURISTIC = "heuristic"


@dataclass(frozen=True)
class SignalMeta:
    key: str
    label: str
    kind: SignalKind
    description: str


SIGNALS: dict[str, SignalMeta] = {
    "cf": SignalMeta(
        key="cf",
        label="Collaborative filtering",
        kind=SignalKind.ML,
        description="Surprise SVD latent-factor model trained on DS1 interactions",
    ),
    "content": SignalMeta(
        key="content",
        label="Content similarity",
        kind=SignalKind.ML,
        description="TF-IDF cosine similarity on DS2+DS3 catalog",
    ),
    "cluster": SignalMeta(
        key="cluster",
        label="Cluster affinity",
        kind=SignalKind.ML,
        description="K-Means user cluster → book affinity from DS1 behaviour",
    ),
    "pop": SignalMeta(
        key="pop",
        label="Popularity",
        kind=SignalKind.HEURISTIC,
        description="Catalog rating_count / db_avg_rating aggregate in SQLite",
    ),
    "genre": SignalMeta(
        key="genre",
        label="Genre overlap",
        kind=SignalKind.HEURISTIC,
        description="Overlap with user genre profile or cluster/global genre priors",
    ),
}

FEATURE_ORDER: list[str] = ["cf", "content", "cluster", "pop", "genre"]


def ml_signal_keys() -> list[str]:
    return [k for k, m in SIGNALS.items() if m.kind == SignalKind.ML]


def heuristic_signal_keys() -> list[str]:
    return [k for k, m in SIGNALS.items() if m.kind == SignalKind.HEURISTIC]


def signal_kind_map() -> dict[str, str]:
    return {k: m.kind.value for k, m in SIGNALS.items()}
