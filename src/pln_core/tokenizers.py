from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from nltk.tokenize import TweetTokenizer

from pln_core.text_utils import TOKEN_PATTERN, fold_text, normalize_text

CUSTOM_TOKENIZER_SOURCE = "custom"
NLTK_TWEET_TOKENIZER_SOURCE = "nltk_tweet"


def tokenize_custom(text: str) -> list[str]:
    """Tokenize text with the project's handcrafted regex tokenizer."""

    folded = fold_text(normalize_text(text))
    return TOKEN_PATTERN.findall(folded)


@lru_cache(maxsize=1)
def _build_tweet_tokenizer() -> TweetTokenizer:
    return TweetTokenizer(
        preserve_case=False,
        reduce_len=True,
        strip_handles=False,
        match_phone_numbers=False,
    )


def tokenize_nltk_tweet(text: str) -> list[str]:
    """Tokenize text with NLTK's TweetTokenizer after project normalization."""

    folded = fold_text(normalize_text(text))
    tokens = _build_tweet_tokenizer().tokenize(folded)
    return [
        token
        for token in tokens
        if token.strip() and TOKEN_PATTERN.fullmatch(token) is not None
    ]


def get_tokenizer(source: str) -> Callable[[str], list[str]]:
    """Return the tokenizer function requested by the CLI or pipeline."""

    if source == CUSTOM_TOKENIZER_SOURCE:
        return tokenize_custom
    if source == NLTK_TWEET_TOKENIZER_SOURCE:
        return tokenize_nltk_tweet
    raise ValueError("Supported tokenizer sources are 'custom' and 'nltk_tweet'.")
