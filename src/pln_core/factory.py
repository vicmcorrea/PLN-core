"""Canonical builder for the production sentiment analyzer.

Both the CLI (``main.py``) and the Streamlit demo import
:func:`build_production_analyzer` so the user-facing pipeline always matches
what the evaluation harness benchmarks under the ``oplexicon_tweet_plus``
configuration. Keeping a single source of truth here avoids drift between the
analyzer described in the report and the one shipped to users.
"""

from __future__ import annotations

from pln_core.eval.analyzers import create_analyzer
from pln_core.pipeline import SymbolicSentimentAnalyzer

PRODUCTION_ANALYZER_NAME = "oplexicon_tweet_plus"
PRODUCTION_ANALYZER_LABEL = (
    "OpLexicon v3.0 + SentiLex-PT 02 + slang/emoji lexicon (tweet-aware)"
)


def build_production_analyzer() -> SymbolicSentimentAnalyzer:
    """Return the production analyzer used by the CLI and Streamlit demo."""

    return create_analyzer(PRODUCTION_ANALYZER_NAME)
