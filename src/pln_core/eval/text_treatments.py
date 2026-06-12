"""Shared text treatments and surface-cue extraction for sentiment benchmarks."""

from __future__ import annotations

import re

TEXT_TREATMENTS = (
    "raw",
    "strip_emoticons_urls",
    "strip_emoticons",
    "strip_urls",
)

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
POSITIVE_EMOTICON_RE = re.compile(
    r"(?i)(?<![\w/])(?:[:;=8][-']?[)DdpP]|x[-']?d|<3)(?!\w)"
)
NEGATIVE_EMOTICON_RE = re.compile(
    r"(?i)(?<![\w/])(?:[:=8][-']?[(cC/\\]|:'\(|[dD]:|</3|<\\3)(?!\w)"
)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
LAUGHTER_RE = re.compile(r"(?i)\b(?:kkk+|rsrs+|haha+|hehe+)\b")
ELONGATED_WORD_RE = re.compile(r"(?i)\b\w*(\w)\1{2,}\w*\b")
EXCLAMATION_RE = re.compile(r"!")
QUESTION_RE = re.compile(r"\?")
WHITESPACE_RE = re.compile(r"\s+")

CUE_PATTERNS = {
    "positive_emoticon": POSITIVE_EMOTICON_RE,
    "negative_emoticon": NEGATIVE_EMOTICON_RE,
    "url": URL_RE,
    "mention": MENTION_RE,
    "hashtag": HASHTAG_RE,
    "laughter": LAUGHTER_RE,
    "elongated_word": ELONGATED_WORD_RE,
    "exclamation": EXCLAMATION_RE,
    "question": QUESTION_RE,
}
EMOTICON_CUES = frozenset({"positive_emoticon", "negative_emoticon"})

LEAKAGE_FEATURE_NAMES = (
    "has_positive_emoticon",
    "has_negative_emoticon",
    "has_url",
)


def text_for_cue_matching(text: str, cue_name: str) -> str:
    """Return the text used for a cue match.

    URLs are removed before emoticon matching so that substrings such as
    ``http://`` are not counted as negative emoticons.
    """

    if cue_name in EMOTICON_CUES:
        return URL_RE.sub(" ", text)
    return text


def has_surface_cue(text: str, cue_name: str) -> bool:
    """Return whether ``text`` contains the named surface cue."""

    try:
        pattern = CUE_PATTERNS[cue_name]
    except KeyError as exc:
        options = ", ".join(sorted(CUE_PATTERNS))
        raise ValueError(f"unknown surface cue '{cue_name}' (expected one of: {options})") from exc
    return bool(pattern.search(text_for_cue_matching(text, cue_name)))


def extract_surface_cues(text: str) -> dict[str, bool]:
    """Extract boolean surface cues used by artifact diagnostics."""

    return {f"has_{cue_name}": has_surface_cue(text, cue_name) for cue_name in CUE_PATTERNS}


def leakage_feature_vector(text: str) -> list[int]:
    """Return the cue-only feature vector requested for leakage diagnostics."""

    cues = extract_surface_cues(text)
    return [int(cues[name]) for name in LEAKAGE_FEATURE_NAMES]


def strip_surface_cues(
    text: str,
    *,
    remove_emoticons: bool,
    remove_urls: bool,
) -> str:
    """Remove selected high-leakage cues and normalize whitespace."""

    cleaned = text
    if remove_urls:
        cleaned = URL_RE.sub(" ", cleaned)
    if remove_emoticons:
        cleaned = POSITIVE_EMOTICON_RE.sub(" ", cleaned)
        cleaned = NEGATIVE_EMOTICON_RE.sub(" ", cleaned)
    return WHITESPACE_RE.sub(" ", cleaned).strip()


def apply_text_treatment(text: str, treatment: str) -> str:
    """Apply a named benchmark text treatment."""

    if treatment in {"raw", "none"}:
        return text
    if treatment == "strip_emoticons_urls":
        return strip_surface_cues(text, remove_emoticons=True, remove_urls=True)
    if treatment == "strip_emoticons":
        return strip_surface_cues(text, remove_emoticons=True, remove_urls=False)
    if treatment == "strip_urls":
        return strip_surface_cues(text, remove_emoticons=False, remove_urls=True)
    options = ", ".join(TEXT_TREATMENTS)
    raise ValueError(f"unknown text treatment '{treatment}' (expected one of: {options})")


def apply_text_treatment_to_many(texts: list[str], treatment: str) -> list[str]:
    """Apply a named text treatment to a list of texts."""

    return [apply_text_treatment(text, treatment) for text in texts]
