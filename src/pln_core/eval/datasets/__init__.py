"""Dataset factory for the evaluation harness.

Each loader lives in its own module and registers itself via
``@DATASET_REGISTRY.register("name")``. The runner reaches them through
:func:`create_dataset`, which is the only public entry point.
"""

from __future__ import annotations

from pln_core.eval.datasets import (  # noqa: F401  (side-effect registers loaders)
    kaggle_tweets,
    sample,
)
from pln_core.eval.datasets.base import EvalDataset, EvalExample
from pln_core.eval.datasets.registry import DATASET_REGISTRY, create_dataset

__all__ = [
    "DATASET_REGISTRY",
    "EvalDataset",
    "EvalExample",
    "create_dataset",
]
