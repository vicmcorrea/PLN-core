"""Canonical builder for the production sentiment analyzer.

The Streamlit app imports :func:`build_production_analyzer`. It uses the
OpLexicon + spaCy lemmas configuration to keep the demo lightweight (no
SentiLex download required). Richer multi-lexicon configurations remain
available in the shared evaluation harness via ``create_analyzer`` for
benchmarking.
"""

from __future__ import annotations

from pln_core.eval.analyzers import create_analyzer
from pln_core.pipeline import SymbolicSentimentAnalyzer

PRODUCTION_ANALYZER_NAME = "oplexicon"
PRODUCTION_ANALYZER_LABEL = "OpLexicon v3.0 + spaCy pt_core_news_sm"


def build_production_analyzer() -> SymbolicSentimentAnalyzer:
    """Return the production analyzer used by the Streamlit demo."""

    return create_analyzer(PRODUCTION_ANALYZER_NAME)
