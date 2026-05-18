"""Evaluation harness for the symbolic sentiment pipeline.

The harness is organized around two small factories (datasets, analyzers) that
are wired together by :func:`pln_core.eval.runner.run_evaluation`. New corpora
or analyzer configurations register themselves via decorators, so they can be
selected from Hydra config files without touching the runner.
"""

from pln_core.eval import analyzers, datasets  # noqa: F401  (register components)
from pln_core.eval.metrics import EvaluationMetrics, compute_metrics
from pln_core.eval.runner import EvaluationReport, run_evaluation

__all__ = [
    "EvaluationMetrics",
    "EvaluationReport",
    "compute_metrics",
    "run_evaluation",
]
