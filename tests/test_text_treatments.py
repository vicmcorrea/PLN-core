"""Tests for sentiment benchmark text treatments and leakage cues."""

from __future__ import annotations

import pytest

from pln_core.eval.text_treatments import (
    LEAKAGE_FEATURE_NAMES,
    apply_text_treatment,
    extract_surface_cues,
    leakage_feature_vector,
)


def test_url_does_not_count_as_negative_emoticon() -> None:
    cues = extract_surface_cues("Veja em http://exemplo.com/noticia")

    assert cues["has_url"] is True
    assert cues["has_negative_emoticon"] is False


def test_extracts_requested_leakage_cues() -> None:
    text = "Adorei :) veja https://exemplo.com"

    cues = extract_surface_cues(text)

    assert cues["has_positive_emoticon"] is True
    assert cues["has_negative_emoticon"] is False
    assert cues["has_url"] is True
    assert LEAKAGE_FEATURE_NAMES == (
        "has_positive_emoticon",
        "has_negative_emoticon",
        "has_url",
    )
    assert leakage_feature_vector(text) == [1, 0, 1]


def test_strip_emoticons_urls_removes_only_requested_cues() -> None:
    text = "Oi @usp #pln adorei :) mas olha http://exemplo.com"

    treated = apply_text_treatment(text, "strip_emoticons_urls")

    assert ":)" not in treated
    assert "http://exemplo.com" not in treated
    assert "@usp" in treated
    assert "#pln" in treated
    assert treated == "Oi @usp #pln adorei mas olha"


def test_unknown_text_treatment_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="unknown text treatment"):
        apply_text_treatment("texto", "misterioso")
