"""Registry holding all evaluation dataset loaders."""

from __future__ import annotations

from pln_core.eval.datasets.base import EvalDataset
from pln_core.eval.registry import Registry

DATASET_REGISTRY: Registry[EvalDataset] = Registry("dataset")


def create_dataset(name: str, **kwargs: object) -> EvalDataset:
    """Build a dataset by name, forwarding ``kwargs`` to the loader."""

    return DATASET_REGISTRY.create(name, **kwargs)
