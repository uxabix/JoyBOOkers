"""Collaborative filtering — Surprise SVD on DS1 ratings."""

from bookrec.ml.collaborative.evaluate import evaluate_svd
from bookrec.ml.collaborative.train import train_svd

__all__ = ["evaluate_svd", "train_svd"]
