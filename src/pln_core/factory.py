"""Canonical builder for the production sentiment analyzer.

Both the CLI (``main.py``) and the Streamlit demo import
:func:`build_production_analyzer`. The demo uses the OpLexicon + spaCy
lemmas configuration to keep the deploy lightweight (no SentiLex download
required). The richer multi-lexicon configurations remain available in the
evaluation harness via ``create_analyzer`` for benchmarking.
"""

from __future__ import annotations

from pln_core.eval.analyzers import create_analyzer
from pln_core.pipeline import SymbolicSentimentAnalyzer

PRODUCTION_ANALYZER_NAME = "oplexicon"
PRODUCTION_ANALYZER_LABEL = "OpLexicon v3.0 + spaCy pt_core_news_sm"


def build_production_analyzer() -> SymbolicSentimentAnalyzer:
    """Return the production analyzer used by the CLI and Streamlit demo."""

    return create_analyzer(PRODUCTION_ANALYZER_NAME)
