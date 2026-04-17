from __future__ import annotations

import csv
import json
from importlib import resources
from pathlib import Path

from pln_core.text_utils import fold_text


def _normalize_lexicon(data: dict[str, float]) -> dict[str, float]:
    return {fold_text(token): float(score) for token, score in data.items()}


def load_seed_lexicon() -> dict[str, float]:
    """Load the bundled starter lexicon shipped with the project."""

    seed_path = resources.files("pln_core.data").joinpath("seed_lexicon.json")
    with seed_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return _normalize_lexicon(data)


def _load_json_lexicon(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("JSON lexicon must be a token->score dictionary.")
    return _normalize_lexicon(data)


def _load_delimited_lexicon(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as file:
        dialect = csv.Sniffer().sniff(file.read(1024), delimiters=",;\t")
        file.seek(0)
        reader = csv.DictReader(file, dialect=dialect)
        rows = list(reader)

    if not rows:
        raise ValueError("Lexicon file is empty.")

    token_key = "token" if "token" in rows[0] else next(iter(rows[0]))
    score_key = "score" if "score" in rows[0] else list(rows[0].keys())[1]
    data = {row[token_key]: float(row[score_key]) for row in rows if row[token_key]}
    return _normalize_lexicon(data)


def load_lexicon(path: str | Path | None = None) -> dict[str, float]:
    """Load a custom lexicon or fall back to the bundled starter lexicon."""

    if path is None:
        return load_seed_lexicon()

    lexicon_path = Path(path)
    suffix = lexicon_path.suffix.lower()

    if suffix == ".json":
        return _load_json_lexicon(lexicon_path)
    if suffix in {".csv", ".tsv"}:
        return _load_delimited_lexicon(lexicon_path)

    raise ValueError("Supported lexicon formats are .json, .csv, and .tsv.")
