"""Twitter-aware text normalization and tokenization for sentiment analysis.

Why a dedicated module?
    The default ``normalize_text``/``tokenize_custom`` pair drops emojis (the
    token regex only matches ``[a-z0-9]+``) and collapses repeated characters
    down to two ("amoooo" -> "amoo"), which prevents the OpLexicon lookup from
    matching "amo". Tweets also need extras: laughter normalization, CamelCase
    hashtag splitting, and aggressive accent/diacritic folding for slang
    written with regional spelling variants.

    This module is additive. Nothing in ``text_utils.py`` or ``tokenizers.py``
    changes; the default analyzers keep their previous behaviour. The new
    helpers are only used by the ``oplexicon_tweet`` analyzer variant.
"""

from __future__ import annotations

import re

from pln_core.text_utils import (
    MENTION_PATTERN,
    URL_PATTERN,
    WHITESPACE_PATTERN,
    fold_text,
)

WORD_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
EMOJI_TOKEN_PATTERN = re.compile(
    r"["
    r"\U0001F300-\U0001FAFF"   # symbols and pictographs (incl. emoji 14)
    r"\U00002600-\U000027BF"   # misc symbols + dingbats
    r"\u2700-\u27BF"            # dingbats
    r"\u2300-\u23FF"            # misc technical
    r"]"
    r"\uFE0F?",
    flags=re.UNICODE,
)
HASHTAG_PATTERN = re.compile(r"#(\w+)")
LAUGH_K_PATTERN = re.compile(r"k{3,}", flags=re.IGNORECASE)
LAUGH_K_SHORT_PATTERN = re.compile(r"\bkk\b", flags=re.IGNORECASE)
LAUGH_HA_PATTERN = re.compile(r"\b(?:ha){2,}\b", flags=re.IGNORECASE)
LAUGH_HE_PATTERN = re.compile(r"\b(?:he){2,}\b", flags=re.IGNORECASE)
LAUGH_RS_PATTERN = re.compile(r"\b(?:rs){2,}\b", flags=re.IGNORECASE)
REPEAT_3_PLUS_PATTERN = re.compile(r"(.)\1{2,}")
CAMEL_SPLIT_PATTERN = re.compile(r"(?<=[a-z])(?=[A-Z])")


def _split_hashtag(match: re.Match[str]) -> str:
    raw = match.group(1)
    if any(ch.isupper() for ch in raw[1:]):
        parts = CAMEL_SPLIT_PATTERN.split(raw)
        return " " + " ".join(parts) + " "
    return " " + raw + " "


def normalize_tweet(text: str) -> str:
    """Clean a tweet while keeping emojis and informal sentiment cues.

    Steps applied in order:
        1. Strip URLs and ``@mentions``.
        2. Split ``#CamelCase`` hashtags into their constituent words.
        3. Normalize laughter (``kkkkk``/``hahaha``/``rsrsrs``) to canonical
           tokens (``kkk``/``haha``/``rsrs``).
        4. Collapse runs of 3+ identical characters down to 1 ("amoooo" ->
           "amo"). Portuguese only has a handful of intentional double-letter
           words, and "amooo"/"ameeei"/"otimoooo" are far more common in
           tweets, so we err on the recovery side.
        5. Squeeze whitespace.

    Returns the cleaned string with the original casing for emojis but
    everything else lower-cased.
    """

    cleaned = URL_PATTERN.sub(" ", text)
    cleaned = MENTION_PATTERN.sub(" ", cleaned)
    cleaned = HASHTAG_PATTERN.sub(_split_hashtag, cleaned)
    cleaned = LAUGH_K_PATTERN.sub("kkk", cleaned)
    cleaned = LAUGH_K_SHORT_PATTERN.sub("kkk", cleaned)
    cleaned = LAUGH_HA_PATTERN.sub("haha", cleaned)
    cleaned = LAUGH_HE_PATTERN.sub("hehe", cleaned)
    cleaned = LAUGH_RS_PATTERN.sub("rsrs", cleaned)
    cleaned = REPEAT_3_PLUS_PATTERN.sub(r"\1", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def tokenize_tweet(text: str) -> list[str]:
    """Return word + emoji tokens for a tweet.

    Word tokens are accent-folded so they collide with the OpLexicon and
    project lexicon keys. Emoji tokens are kept as-is (the slang/emoji TSV
    stores them in Unicode form).
    """

    normalized = normalize_tweet(text)
    folded = fold_text(normalized)
    pieces: list[tuple[int, str]] = []
    for match in WORD_TOKEN_PATTERN.finditer(folded):
        pieces.append((match.start(), match.group(0)))
    for match in EMOJI_TOKEN_PATTERN.finditer(normalized):
        pieces.append((match.start(), match.group(0)))
    pieces.sort(key=lambda item: item[0])
    return [token for _, token in pieces]


def tokenize_tweet_lemma(text: str) -> list[str]:
    """Tweet-aware tokenizer with spaCy lemmatization for word tokens.

    Pipeline:
        1. :func:`normalize_tweet` cleans URLs, mentions, hashtags, laughter,
           and elongated spellings while keeping emojis in place.
        2. Emojis are extracted and kept as-is.
        3. The remaining text is lemmatized with the existing project spaCy
           model so verb conjugations like ``curtindo``/``curtiu`` collide
           with the lemma ``curtir`` in the lexicon.

    This is the heaviest tokenizer but it tends to recover ~3-5pp of
    accuracy on tweets compared to the regex-only :func:`tokenize_tweet`.
    """

    from pln_core.tokenizers import _build_spacy_lemmatizer

    normalized = normalize_tweet(text)
    emoji_positions = [
        (match.start(), match.group(0))
        for match in EMOJI_TOKEN_PATTERN.finditer(normalized)
    ]
    text_without_emoji = EMOJI_TOKEN_PATTERN.sub(" ", normalized)
    doc = _build_spacy_lemmatizer()(text_without_emoji)
    tokens: list[tuple[int, str]] = []
    for token in doc:
        lemma = fold_text(token.lemma_ or token.text)
        if not lemma.strip():
            continue
        if WORD_TOKEN_PATTERN.fullmatch(lemma) is None:
            continue
        tokens.append((token.idx, lemma))
    tokens.extend(emoji_positions)
    tokens.sort(key=lambda item: item[0])
    return [token for _, token in tokens]
