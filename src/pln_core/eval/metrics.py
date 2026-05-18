"""Metrics computed by the evaluation runner.

We keep the numbers self contained so the runner does not depend on
scikit-learn for anything beyond the confusion matrix and macro-F1.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.metrics import classification_report, confusion_matrix, f1_score

from pln_core.eval.datasets.base import VALID_LABELS


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Summary statistics for a single analyzer/dataset combination."""

    total: int
    correct: int
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    support: dict[str, int]
    confusion: dict[tuple[str, str], int]
    classification_report: str

    def confusion_matrix_rows(self) -> list[tuple[str, list[int]]]:
        """Return ``(true_label, [count_for_each_pred_label])`` rows."""

        rows: list[tuple[str, list[int]]] = []
        for true_label in VALID_LABELS:
            counts = [self.confusion.get((true_label, pred), 0) for pred in VALID_LABELS]
            rows.append((true_label, counts))
        return rows


def compute_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> EvaluationMetrics:
    """Compute accuracy, macro F1 and the confusion matrix for the run."""

    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    if not expected:
        raise ValueError("cannot compute metrics on empty result set")

    total = len(expected)
    correct = sum(1 for exp, pred in zip(expected, predicted, strict=False) if exp == pred)

    accuracy = correct / total
    labels = list(VALID_LABELS)

    macro_f1 = float(
        f1_score(expected, predicted, labels=labels, average="macro", zero_division=0)
    )
    per_class = f1_score(expected, predicted, labels=labels, average=None, zero_division=0)
    per_class_f1 = {label: float(value) for label, value in zip(labels, per_class, strict=True)}

    matrix = confusion_matrix(expected, predicted, labels=labels)
    confusion = {
        (true_label, pred_label): int(matrix[i, j])
        for i, true_label in enumerate(labels)
        for j, pred_label in enumerate(labels)
    }
    support = dict(Counter(expected))
    for label in labels:
        support.setdefault(label, 0)

    report = classification_report(
        expected,
        predicted,
        labels=labels,
        digits=3,
        zero_division=0,
    )

    return EvaluationMetrics(
        total=total,
        correct=correct,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        support=support,
        confusion=confusion,
        classification_report=report,
    )
