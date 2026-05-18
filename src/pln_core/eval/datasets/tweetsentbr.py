"""TweetSentBR (few-shot) loader from the HuggingFace Hub.

Wraps the ``eduagarcia/tweetsentbr_fewshot`` dataset which exposes the
official test split (2,010 tweets, 3 classes) of the original TweetSentBR
corpus by Brum and Volpe Nunes (LREC 2018).
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from pln_core.eval.datasets.base import (
    EvalDataset,
    EvalExample,
    stratified_sample,
)
from pln_core.eval.datasets.registry import DATASET_REGISTRY

HF_DATASET_ID = "eduagarcia/tweetsentbr_fewshot"

LABEL_MAP: dict[str, str] = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
}


def _normalize_label(raw: str) -> str:
    key = raw.strip().lower()
    try:
        return LABEL_MAP[key]
    except KeyError as exc:
        raise ValueError(f"unknown TweetSentBR label '{raw}'") from exc


def _iter_rows(rows: Iterable[dict[str, object]]) -> Iterable[EvalExample]:
    for row in rows:
        sentence = str(row.get("sentence", "")).strip()
        if not sentence:
            continue
        label = _normalize_label(str(row["label"]))
        yield EvalExample(text=sentence, label=label)


@DATASET_REGISTRY.register("tweetsentbr")
def load_tweetsentbr(
    split: str = "test",
    max_examples: int | None = None,
    per_class: int | None = None,
    balanced: bool = False,
    seed: int = 42,
    cache_dir: str | None = None,
) -> EvalDataset:
    """Load the TweetSentBR few-shot corpus from the HuggingFace Hub.

    Args:
        split: Which split to use. Defaults to ``"test"`` (2,010 tweets).
        max_examples: Optional cap on the total number of examples (random
            shuffle with ``seed``). Ignored when ``balanced`` or ``per_class``
            is set.
        per_class: When provided (or when ``balanced`` is True without a
            specific value), sample this many examples per label using a
            stratified shuffle. Pass ``0`` to keep everything.
        balanced: Convenience flag; when True and ``per_class`` is None, sample
            670 tweets per class (gives the ~2,010 size of the natural test
            split but balanced).
        seed: Seed for any sampling.
        cache_dir: Optional cache directory passed to ``datasets.load_dataset``.

    Returns:
        EvalDataset with ``text`` and lowercase label fields.
    """

    from datasets import load_dataset

    raw = load_dataset(
        HF_DATASET_ID,
        split=split,
        cache_dir=cache_dir or os.environ.get("HF_DATASETS_CACHE"),
    )

    all_examples = list(_iter_rows(raw))

    if balanced and per_class is None:
        per_class = 670

    if per_class and per_class > 0:
        selected = stratified_sample(all_examples, per_class=per_class, seed=seed)
        suffix = f"@{per_class}per_class"
    elif max_examples is not None and max_examples > 0 and max_examples < len(all_examples):
        rng_view = raw.shuffle(seed=seed).select(range(max_examples))
        selected = list(_iter_rows(rng_view))
        suffix = f"@{max_examples}"
    else:
        selected = all_examples
        suffix = ""

    return EvalDataset(
        name=f"tweetsentbr[{split}]{suffix}",
        description=(
            "TweetSentBR (Brum and Volpe Nunes, LREC 2018) - few-shot release "
            f"via {HF_DATASET_ID}, split={split}"
        ),
        examples=tuple(selected),
    )
