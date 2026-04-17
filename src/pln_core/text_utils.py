from __future__ import annotations

import re
import unicodedata

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")
WHITESPACE_PATTERN = re.compile(r"\s+")


def fold_text(text: str) -> str:
    """Lowercase text and strip accents for lexicon matching."""

    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_text(text: str) -> str:
    """Clean noisy social text while preserving readable output."""

    cleaned = URL_PATTERN.sub(" ", text)
    cleaned = MENTION_PATTERN.sub(" ", cleaned)
    cleaned = HASHTAG_PATTERN.sub(r"\1", cleaned)
    cleaned = REPEATED_CHAR_PATTERN.sub(r"\1\1", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def tokenize(text: str) -> list[str]:
    """Convert text into normalized alphanumeric tokens."""

    folded = fold_text(normalize_text(text))
    return TOKEN_PATTERN.findall(folded)
