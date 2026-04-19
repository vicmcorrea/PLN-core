"""Music recommendation lookups based on the analyzer's sentiment label and score.

The catalog lives in ``data/recommendations.json`` and is a list of song entries
with ``sentiment`` (matching the analyzer labels) and ``valence`` (a float in
``[-1, 1]`` describing the song's mood intensity). Songs are ranked by how
close their valence is to the analyzer score, after the score is clipped to
``[-1, 1]``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "recommendations.json"
)


@dataclass(frozen=True, slots=True)
class Song:
    title: str
    artist: str
    youtube_id: str
    sentiment: str
    valence: float
    tags: tuple[str, ...] = ()

    @property
    def youtube_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.youtube_id}"

    @property
    def search_url(self) -> str:
        from urllib.parse import quote_plus

        query = quote_plus(f"{self.artist} {self.title}")
        return f"https://www.youtube.com/results?search_query={query}"


def _coerce_song(entry: dict) -> Song:
    return Song(
        title=str(entry["title"]),
        artist=str(entry["artist"]),
        youtube_id=str(entry["youtube_id"]),
        sentiment=str(entry["sentiment"]),
        valence=float(entry["valence"]),
        tags=tuple(entry.get("tags", ())),
    )


def load_catalog(path: Path | None = None) -> tuple[Song, ...]:
    target = path or DEFAULT_CATALOG_PATH
    raw = json.loads(target.read_text(encoding="utf-8"))
    return tuple(_coerce_song(entry) for entry in raw)


_CATALOG_CACHE: dict[tuple[Path, float], tuple[Song, ...]] = {}


def _default_catalog() -> tuple[Song, ...]:
    target = DEFAULT_CATALOG_PATH
    key = (target, target.stat().st_mtime)
    cached = _CATALOG_CACHE.get(key)
    if cached is not None:
        return cached
    _CATALOG_CACHE.clear()
    catalog = load_catalog(target)
    _CATALOG_CACHE[key] = catalog
    return catalog


def recommend_ranked(
    label: str,
    score: float,
    catalog: Iterable[Song] | None = None,
) -> tuple[Song, ...]:
    """All catalog songs matching ``label``, sorted by valence proximity to ``score``."""

    pool = tuple(catalog) if catalog is not None else _default_catalog()
    candidates = [song for song in pool if song.sentiment == label]
    if not candidates:
        return ()

    target = max(-1.0, min(1.0, float(score)))
    candidates.sort(key=lambda song: abs(song.valence - target))
    return tuple(candidates)


def recommend(
    label: str,
    score: float,
    k: int = 3,
    catalog: Iterable[Song] | None = None,
) -> tuple[Song, ...]:
    """Return up to ``k`` songs that match ``label`` ranked by valence proximity."""

    ranked = recommend_ranked(label, score, catalog=catalog)
    return ranked[: max(0, k)]
