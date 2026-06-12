"""Canonical builder for the production sentiment analyzer.

The Streamlit app imports :func:`build_production_analyzer`. The active Etapa 1
baseline is intentionally the same lightweight configuration used in the
Kaggle benchmark: OpLexicon v3.0 plus the project's regex tokenizer.
"""

from __future__ import annotations

from pln_core.eval.analyzers import create_analyzer
from pln_core.pipeline import SymbolicSentimentAnalyzer

PRODUCTION_ANALYZER_NAME = "oplexicon_regex"
PRODUCTION_ANALYZER_LABEL = "OpLexicon v3.0 + regex tokenizer"


def build_production_analyzer() -> SymbolicSentimentAnalyzer:
    """Return the production analyzer used by the Streamlit demo."""

    return create_analyzer(PRODUCTION_ANALYZER_NAME)
