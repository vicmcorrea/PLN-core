"""Run an analyzer over a dataset and collect metrics."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from pln_core.eval.analyzers import create_analyzer
from pln_core.eval.datasets import EvalDataset, EvalExample, create_dataset
from pln_core.eval.metrics import EvaluationMetrics, compute_metrics
from pln_core.pipeline import SymbolicSentimentAnalyzer


@dataclass(frozen=True, slots=True)
class CasePrediction:
    text: str
    expected: str
    predicted: str
    score: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    analyzer: str
    dataset: str
    dataset_description: str
    elapsed_seconds: float
    metrics: EvaluationMetrics
    predictions: tuple[CasePrediction, ...]

    def as_dict(self, include_predictions: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "analyzer": self.analyzer,
            "dataset": self.dataset,
            "dataset_description": self.dataset_description,
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": {
                "total": self.metrics.total,
                "correct": self.metrics.correct,
                "accuracy": self.metrics.accuracy,
                "macro_f1": self.metrics.macro_f1,
                "per_class_f1": self.metrics.per_class_f1,
                "support": self.metrics.support,
                "confusion": {
                    f"{true}->{pred}": count
                    for (true, pred), count in self.metrics.confusion.items()
                },
                "classification_report": self.metrics.classification_report,
            },
        }
        if include_predictions:
            payload["predictions"] = [asdict(case) for case in self.predictions]
        return payload


def _predict(
    analyzer: SymbolicSentimentAnalyzer,
    examples: Iterable[EvalExample],
) -> tuple[CasePrediction, ...]:
    predictions: list[CasePrediction] = []
    for example in examples:
        result = analyzer.analyze(example.text)
        predictions.append(
            CasePrediction(
                text=example.text,
                expected=example.label,
                predicted=result.label,
                score=result.score,
            )
        )
    return tuple(predictions)


def run_evaluation(
    analyzer_name: str,
    dataset_name: str,
    dataset_kwargs: dict[str, object] | None = None,
    analyzer_kwargs: dict[str, object] | None = None,
) -> EvaluationReport:
    """Build the analyzer and dataset by name, run them, and collect metrics."""

    dataset: EvalDataset = create_dataset(dataset_name, **(dataset_kwargs or {}))
    analyzer = create_analyzer(analyzer_name, **(analyzer_kwargs or {}))

    start = time.perf_counter()
    predictions = _predict(analyzer, dataset.examples)
    elapsed = time.perf_counter() - start

    metrics = compute_metrics(
        expected=[case.expected for case in predictions],
        predicted=[case.predicted for case in predictions],
    )
    return EvaluationReport(
        analyzer=analyzer_name,
        dataset=dataset.name,
        dataset_description=dataset.description,
        elapsed_seconds=elapsed,
        metrics=metrics,
        predictions=predictions,
    )


def save_report_json(
    report: EvaluationReport,
    destination: Path,
    include_predictions: bool = False,
) -> Path:
    """Persist a report as JSON, creating parent directories if needed."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(include_predictions=include_predictions), handle, indent=2)
    return destination
