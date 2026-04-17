from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import spacy

from pln_core.text_utils import TOKEN_PATTERN, fold_text, normalize_text

CUSTOM_TOKENIZER_SOURCE = "custom"
SPACY_PT_TOKENIZER_SOURCE = "spacy_pt"


def tokenize_custom(text: str) -> list[str]:
    """Tokenize text with the project's handcrafted regex tokenizer."""

    folded = fold_text(normalize_text(text))
    return TOKEN_PATTERN.findall(folded)


@lru_cache(maxsize=1)
def _build_spacy_tokenizer():
    return spacy.blank("pt").tokenizer


def tokenize_spacy_pt(text: str) -> list[str]:
    """Tokenize text with spaCy's blank Portuguese tokenizer."""

    folded = fold_text(normalize_text(text))
    doc = _build_spacy_tokenizer()(folded)
    return [
        token.text
        for token in doc
        if token.text.strip() and TOKEN_PATTERN.fullmatch(token.text) is not None
    ]


def get_tokenizer(source: str) -> Callable[[str], list[str]]:
    """Return the tokenizer function requested by the CLI or pipeline."""

    if source == CUSTOM_TOKENIZER_SOURCE:
        return tokenize_custom
    if source == SPACY_PT_TOKENIZER_SOURCE:
        return tokenize_spacy_pt
    raise ValueError("Supported tokenizer sources are 'custom' and 'spacy_pt'.")
