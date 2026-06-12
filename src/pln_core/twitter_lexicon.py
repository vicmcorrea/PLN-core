"""Curated slang and emoji sentiment lexicon for Brazilian Portuguese tweets.

Each row maps a single ``token`` (lower-case, already accent-folded for words)
to a polarity score in [-1, +1]. The format mirrors the OpLexicon TSV used by
the rest of the project so the same delimited loader can read it.

Polarity convention:
    +1.0  strong positive  (e.g., "amei", "perfeito", "❤️")
    +0.5  mild positive
    -0.5  mild negative
    -1.0  strong negative  (e.g., "pessimo", "horrivel", "😡")

Sources:
    - Brazilian Portuguese social-media usage observed during project
      development and corpus inspection.
    - Common emoji polarity from Novak et al. (2015) "Sentiment of Emojis",
      restricted to the high-confidence subset.
"""

import csv
from importlib import resources

from pln_core.text_utils import fold_text


def load_twitter_extras() -> dict[str, float]:
    """Load the bundled slang + emoji lexicon as a ``{token: score}`` mapping.

    Word tokens are accent-folded with :func:`fold_text` so they match the
    folded keys produced by the project tokenizers. Emoji tokens are kept
    verbatim because they are stored as Unicode characters.
    """

    path = resources.files("pln_core.data").joinpath("slang_emoji_ptbr.tsv")
    extras: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            token = (row.get("token") or "").strip()
            if not token:
                continue
            try:
                score = float(row["score"])
            except (KeyError, ValueError):
                continue
            key = token if _is_emoji_token(token) else fold_text(token)
            extras[key] = score
    return extras


def _is_emoji_token(token: str) -> bool:
    """Heuristic check: token is emoji if it has no ascii letters/digits."""

    return not any(ch.isascii() and ch.isalnum() for ch in token)
