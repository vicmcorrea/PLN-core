"""Base types and shared helpers for evaluation dataset loaders."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

VALID_LABELS = ("positive", "negative", "neutral")


@dataclass(frozen=True, slots=True)
class EvalExample:
    """A single labeled sentence used during evaluation."""

    text: str
    label: str

    def __post_init__(self) -> None:
        if self.label not in VALID_LABELS:
            raise ValueError(
                f"invalid label '{self.label}' (expected one of {VALID_LABELS})"
            )


@dataclass(frozen=True, slots=True)
class EvalDataset:
    """A named, ordered collection of evaluation examples."""

    name: str
    description: str
    examples: Sequence[EvalExample]

    def __len__(self) -> int:
        return len(self.examples)


def rating_to_3class(rating: int | str | None) -> str | None:
    """Map a 1-5 star rating to ``negative``/``neutral``/``positive``.

    Returns ``None`` if the rating is missing or outside the 1-5 range.
    """

    try:
        score = int(rating)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if score in (1, 2):
        return "negative"
    if score == 3:
        return "neutral"
    if score in (4, 5):
        return "positive"
    return None


def stratified_sample(
    examples: Iterable[EvalExample],
    per_class: int,
    seed: int,
) -> list[EvalExample]:
    """Pick ``per_class`` items for each label using ``random.Random(seed)``.

    If a class has fewer than ``per_class`` examples, all of them are kept.
    The final list is shuffled with the same seed so order is reproducible.
    """

    by_label: dict[str, list[EvalExample]] = {}
    for example in examples:
        by_label.setdefault(example.label, []).append(example)

    rng = random.Random(seed)
    selected: list[EvalExample] = []
    for items in by_label.values():
        rng.shuffle(items)
        selected.extend(items[:per_class])
    rng.shuffle(selected)
    return selected
