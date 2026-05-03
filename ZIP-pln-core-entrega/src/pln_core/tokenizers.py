from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import spacy

from pln_core.text_utils import TOKEN_PATTERN, fold_text, normalize_text

CUSTOM_TOKENIZER_SOURCE = "custom"
SPACY_PT_TOKENIZER_SOURCE = "spacy_pt"
SPACY_PT_LEMMATIZER_SOURCE = "spacy_pt_lemma"
LEMMA_OVERRIDES = {
    "amei": "amar",
}
IGNORED_LEMMAS = {"ser", "estar", "ter", "haver", "ir"}


def tokenize_custom(text: str) -> list[str]:
    """Tokenize text with the project's handcrafted regex tokenizer."""

    folded = fold_text(normalize_text(text))
    return TOKEN_PATTERN.findall(folded)


@lru_cache(maxsize=1)
def _build_spacy_tokenizer():
    return spacy.blank("pt").tokenizer


@lru_cache(maxsize=1)
def _build_spacy_lemmatizer():
    nlp = spacy.load("pt_core_news_sm")
    ruler = nlp.get_pipe("attribute_ruler")
    for token, lemma in LEMMA_OVERRIDES.items():
        ruler.add(patterns=[[{"LOWER": token}]], attrs={"LEMMA": lemma})
    return nlp


def tokenize_spacy_pt(text: str) -> list[str]:
    """Tokenize text with spaCy's blank Portuguese tokenizer."""

    folded = fold_text(normalize_text(text))
    doc = _build_spacy_tokenizer()(folded)
    return [
        token.text
        for token in doc
        if token.text.strip() and TOKEN_PATTERN.fullmatch(token.text) is not None
    ]


def tokenize_spacy_pt_lemmas(text: str) -> list[str]:
    """Tokenize text with spaCy and return normalized Portuguese lemmas."""

    doc = _build_spacy_lemmatizer()(normalize_text(text))
    lemmas: list[str] = []
    for token in doc:
        lemma = fold_text(token.lemma_ or token.text)
        if (
            lemma.strip()
            and lemma not in IGNORED_LEMMAS
            and TOKEN_PATTERN.fullmatch(lemma) is not None
        ):
            lemmas.append(lemma)
    return lemmas


def get_tokenizer(source: str) -> Callable[[str], list[str]]:
    """Return the tokenizer function requested by the CLI or pipeline."""

    if source == CUSTOM_TOKENIZER_SOURCE:
        return tokenize_custom
    if source == SPACY_PT_TOKENIZER_SOURCE:
        return tokenize_spacy_pt
    if source == SPACY_PT_LEMMATIZER_SOURCE:
        return tokenize_spacy_pt_lemmas
    raise ValueError(
        "Supported tokenizer sources are 'custom', 'spacy_pt', and 'spacy_pt_lemma'."
    )
