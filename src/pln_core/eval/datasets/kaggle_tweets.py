"""Loader for the Kaggle Portuguese tweets sentiment corpus."""

from __future__ import annotations

import csv
import random
from collections.abc import Iterable
from pathlib import Path

from pln_core.eval.datasets.base import EvalDataset, EvalExample, stratified_sample
from pln_core.eval.datasets.registry import DATASET_REGISTRY

DEFAULT_SOURCE_DIR = Path("data/raw/portuguese-tweets-for-sentiment-analysis")
DEFAULT_SPLIT_FILES = {
    "train": Path("TrainingDatasets/Train3Classes.csv"),
    "test": Path("TestDatasets/Test3classes.csv"),
}

LABEL_MAP = {
    "0": "negative",
    "1": "positive",
    "2": "neutral",
    "negative": "negative",
    "negativo": "negative",
    "positive": "positive",
    "positivo": "positive",
    "neutral": "neutral",
    "neutro": "neutral",
}


def _normalize_label(raw: object) -> str:
    key = str(raw).strip().lower()
    try:
        return LABEL_MAP[key]
    except KeyError as exc:
        raise ValueError(f"unknown Kaggle tweet sentiment label '{raw}'") from exc


def _resolve_path(split: str, source_dir: str | None, file_path: str | None) -> Path:
    if file_path:
        path = Path(file_path)
    else:
        try:
            split_file = DEFAULT_SPLIT_FILES[split]
        except KeyError as exc:
            options = ", ".join(sorted(DEFAULT_SPLIT_FILES))
            raise ValueError(f"unknown split '{split}' (expected one of: {options})") from exc
        path = Path(source_dir or DEFAULT_SOURCE_DIR) / split_file

    if not path.exists():
        raise FileNotFoundError(
            "Kaggle tweets dataset file not found: "
            f"{path}. Download augustop/portuguese-tweets-for-sentiment-analysis "
            "and place it under data/raw/portuguese-tweets-for-sentiment-analysis."
        )
    return path


def _read_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        yield from reader


def _first_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return value
    return ""


def _iter_examples(path: Path) -> Iterable[EvalExample]:
    for row in _read_rows(path):
        text = _first_value(row, "tweet_text", "text", "sentence", "tweet").strip()
        if not text:
            continue
        raw_label = _first_value(row, "sentiment", "label", "class")
        yield EvalExample(text=text, label=_normalize_label(raw_label))


@DATASET_REGISTRY.register("kaggle_tweets")
def load_kaggle_tweets(
    split: str = "test",
    source_dir: str | None = None,
    file_path: str | None = None,
    max_examples: int | None = None,
    per_class: int | None = None,
    balanced: bool = False,
    seed: int = 42,
) -> EvalDataset:
    """Load the Kaggle Portuguese tweets corpus from local CSV files."""

    path = _resolve_path(split=split, source_dir=source_dir, file_path=file_path)
    all_examples = list(_iter_examples(path))

    if balanced and per_class is None:
        counts = {
            label: sum(1 for example in all_examples if example.label == label)
            for label in ("positive", "negative", "neutral")
        }
        per_class = min(counts.values())

    if per_class and per_class > 0:
        selected = stratified_sample(all_examples, per_class=per_class, seed=seed)
        suffix = f"@{per_class}per_class"
    elif max_examples is not None and 0 < max_examples < len(all_examples):
        selected = all_examples[:]
        random.Random(seed).shuffle(selected)
        selected = selected[:max_examples]
        suffix = f"@{max_examples}"
    else:
        selected = all_examples
        suffix = ""

    return EvalDataset(
        name=f"kaggle_tweets[{split}]{suffix}",
        description=(
            "Kaggle Portuguese Tweets for Sentiment Analysis "
            f"(augustop), split={split}, file={path}"
        ),
        examples=tuple(selected),
    )
