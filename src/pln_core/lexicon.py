from __future__ import annotations

import contextlib
import csv
import json
from collections import defaultdict
from importlib import resources
from pathlib import Path
from urllib import request
from urllib.error import URLError

from pln_core.text_utils import fold_text

SEED_LEXICON_SOURCE = "seed"
OPLEXICON_LEXICON_SOURCE = "oplexicon"
OPLEXICON_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/sillasgonzaga/lexiconPT/master/"
    "data-raw/oplexicon_v3.0/lexico_v3.0.txt"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPLEXICON_CACHE_PATH = PROJECT_ROOT / "data" / "external" / "oplexicon_v3.0.txt"


class LexiconDownloadError(RuntimeError):
    """Raised when an external lexicon cannot be downloaded."""


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


def ensure_oplexicon_file(path: str | Path | None = None) -> Path:
    """Download OpLexicon v3.0 to a local cache if it is not already present."""

    target_path = Path(path) if path is not None else OPLEXICON_CACHE_PATH
    if target_path.exists():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            contextlib.closing(request.urlopen(OPLEXICON_DOWNLOAD_URL, timeout=30)) as response,
            target_path.open("wb") as file,
        ):
            file.write(response.read())
    except (OSError, URLError) as exc:
        raise LexiconDownloadError("could not download oplexicon v3.0") from exc

    return target_path


def load_oplexicon(path: str | Path | None = None) -> dict[str, float]:
    """Load OpLexicon v3.0 and collapse duplicate entries by averaged polarity."""

    lexicon_path = ensure_oplexicon_file(path)
    scores_by_token: dict[str, list[float]] = defaultdict(list)

    with lexicon_path.open("r", encoding="utf-8", errors="ignore") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 3:
                continue

            token = fold_text(row[0].strip())
            if not token:
                continue

            try:
                polarity = float(row[2])
            except ValueError:
                continue

            scores_by_token[token].append(polarity)

    return {
        token: round(sum(values) / len(values), 3)
        for token, values in scores_by_token.items()
    }


def load_lexicon(
    path: str | Path | None = None,
    source: str = SEED_LEXICON_SOURCE,
) -> dict[str, float]:
    """Load the built-in seed lexicon, OpLexicon, or a custom lexicon path."""

    if path is None:
        if source == SEED_LEXICON_SOURCE:
            return load_seed_lexicon()
        if source == OPLEXICON_LEXICON_SOURCE:
            return load_oplexicon()
        raise ValueError("Supported lexicon sources are 'seed' and 'oplexicon'.")

    lexicon_path = Path(path)
    suffix = lexicon_path.suffix.lower()

    if lexicon_path.name == "lexico_v3.0.txt":
        return load_oplexicon(lexicon_path)
    if suffix == ".json":
        return _load_json_lexicon(lexicon_path)
    if suffix in {".csv", ".tsv"}:
        return _load_delimited_lexicon(lexicon_path)
    if suffix == ".txt":
        return load_oplexicon(lexicon_path)

    raise ValueError("Supported lexicon formats are .json, .csv, .tsv, and OpLexicon .txt.")
