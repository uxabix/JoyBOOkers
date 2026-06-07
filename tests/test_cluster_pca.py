"""Tests for online PCA projection of user rating profiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from app.ml.cluster_pca import ClusterPcaProjector
from app.ml.user_clustering import UserClusteringEngine
from bookrec.ml.io import save_joblib


def test_cluster_pca_projector_roundtrip(tmp_path: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    X = np.random.randn(40, 7).astype(np.float32)
    pca.fit(X)
    model_path = tmp_path / "pca_viz.joblib"
    save_joblib(pca, model_path)

    projector = ClusterPcaProjector(model_path)
    assert projector.load()

    engine = UserClusteringEngine(
        tmp_path / "missing.joblib",
        tmp_path,
        tmp_path / "missing.json",
    )
    engine._mu = np.zeros(7, dtype=np.float32)
    engine._sigma = np.ones(7, dtype=np.float32)
    engine._model = object()

    scores = [5.0, 4.0, 5.0, 3.0, 5.0]
    point = projector.highlight_point(engine, scores, label="Test", kind="user", cluster_id=1)
    assert point is not None
    assert "x" in point and "y" in point
