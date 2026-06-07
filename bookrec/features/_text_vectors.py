"""Sparse bag-of-words helpers (memory-safe for ~100k documents)."""

from __future__ import annotations

import re

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(str(text).lower())


def build_sparse_bow(
    texts: list[str],
    max_features: int = 5000,
) -> tuple[csr_matrix, list[str]]:
    """Document-frequency vocabulary + CSR count matrix, L2-normalized rows."""
    vocab: dict[str, int] = {}
    for text in texts:
        for t in set(tokenize(text)):
            vocab[t] = vocab.get(t, 0) + 1
    sorted_terms = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))[:max_features]
    term_to_idx = {t: i for i, (t, _) in enumerate(sorted_terms)}
    vocabulary = list(term_to_idx.keys())

    n_docs = len(texts)
    n_terms = len(term_to_idx)
    mat = lil_matrix((n_docs, n_terms), dtype=np.float32)
    for i, text in enumerate(texts):
        for t in tokenize(text):
            j = term_to_idx.get(t)
            if j is not None:
                mat[i, j] += 1.0

    csr = csr_matrix(mat)
    # L2 normalize each row
    norms = np.sqrt(csr.multiply(csr).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    inv = 1.0 / norms
    csr = csr.multiply(inv[:, np.newaxis])
    return csr, vocabulary
