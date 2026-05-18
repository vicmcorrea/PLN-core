"""Factory for sentiment analyzer configurations used during evaluation."""

from __future__ import annotations

from pln_core.eval.registry import Registry
from pln_core.lexicon import OPLEXICON_LEXICON_SOURCE, load_lexicon
from pln_core.pipeline import SymbolicSentimentAnalyzer
from pln_core.sentilex import load_sentilex_flex
from pln_core.tokenizers import tokenize_spacy_pt_lemmas
from pln_core.twitter_lexicon import load_twitter_extras
from pln_core.twitter_norm import tokenize_tweet, tokenize_tweet_lemma

ANALYZER_REGISTRY: Registry[SymbolicSentimentAnalyzer] = Registry("analyzer")


def create_analyzer(name: str, **kwargs: object) -> SymbolicSentimentAnalyzer:
    """Build a configured analyzer by name."""

    return ANALYZER_REGISTRY.create(name, **kwargs)


@ANALYZER_REGISTRY.register("seed")
def _build_seed_analyzer(**kwargs: object) -> SymbolicSentimentAnalyzer:
    """Didactic baseline: ~30 word seed lexicon, regex tokenizer."""

    return SymbolicSentimentAnalyzer(**kwargs)  # type: ignore[arg-type]


@ANALYZER_REGISTRY.register("oplexicon")
def _build_oplexicon_analyzer(**kwargs: object) -> SymbolicSentimentAnalyzer:
    """Production stack: OpLexicon v3.0 + spaCy ``pt_core_news_sm`` lemmas.

    Accepts the same threshold and rule toggles as
    :class:`SymbolicSentimentAnalyzer`, forwarded via ``kwargs``. Used by the
    ablation sweep to disable individual rules or move the decision threshold.
    """

    lexicon = load_lexicon(source=OPLEXICON_LEXICON_SOURCE)
    return SymbolicSentimentAnalyzer(
        lexicon=lexicon,
        tokenizer=tokenize_spacy_pt_lemmas,
        **kwargs,  # type: ignore[arg-type]
    )


@ANALYZER_REGISTRY.register("oplexicon_regex")
def _build_oplexicon_regex_analyzer(**kwargs: object) -> SymbolicSentimentAnalyzer:
    """OpLexicon with the simple regex tokenizer (no spaCy lemmatization)."""

    lexicon = load_lexicon(source=OPLEXICON_LEXICON_SOURCE)
    return SymbolicSentimentAnalyzer(lexicon=lexicon, **kwargs)  # type: ignore[arg-type]


def _merge_lexicons(*sources: dict[str, float]) -> dict[str, float]:
    """Right-most source wins on key collisions (later sources override)."""

    merged: dict[str, float] = {}
    for source in sources:
        merged.update(source)
    return merged


@ANALYZER_REGISTRY.register("oplexicon_tweet")
def _build_oplexicon_tweet_analyzer(
    positive_threshold: float = 0.5,
    negative_threshold: float = -0.5,
    **kwargs: object,
) -> SymbolicSentimentAnalyzer:
    """OpLexicon plus a curated PT-BR slang + emoji extension.

    Differences from ``oplexicon_regex``:
        * Lexicon is the union of OpLexicon v3.0 with ~180 hand-curated tweet
          tokens (slang, internet abbreviations, laughter, emoji polarities).
          The slang/emoji entries override any conflicting OpLexicon entry,
          because the latter sometimes has stale or context-free polarities for
          informal terms.
        * Tokenizer is :func:`tokenize_tweet`, which keeps emojis and
          normalizes laughter and elongated spellings ("amooo" -> "amo").
        * Default decision thresholds are loosened to ``\u00b10.5`` to undo the
          neutral bias caused by tweets accumulating fewer matched terms than
          longer reviews.

    All threshold arguments and rule toggles inherited from
    :class:`SymbolicSentimentAnalyzer` remain configurable via ``kwargs``.
    """

    lexicon = _merge_lexicons(
        load_lexicon(source=OPLEXICON_LEXICON_SOURCE),
        load_twitter_extras(),
    )
    return SymbolicSentimentAnalyzer(
        lexicon=lexicon,
        tokenizer=tokenize_tweet,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        **kwargs,  # type: ignore[arg-type]
    )


@ANALYZER_REGISTRY.register("oplexicon_tweet_plus")
def _build_oplexicon_tweet_plus_analyzer(
    positive_threshold: float = 0.5,
    negative_threshold: float = -0.5,
    sentilex_weight: float = 0.6,
    **kwargs: object,
) -> SymbolicSentimentAnalyzer:
    """``oplexicon_tweet`` fused with SentiLex-PT 02 (multi-lexicon).

    Merge order (later overrides earlier on key collisions):
        1. SentiLex-PT 02 inflected forms, scaled by ``sentilex_weight`` so
           its uniform ``+/-1`` polarities do not dominate OpLexicon's more
           graded scores.
        2. OpLexicon v3.0.
        3. Curated slang/emoji extras.

    Tokenizer and rule engine are the same tweet-aware stack from
    ``oplexicon_tweet``. The merge expands lexical coverage from ~31k
    (OpLexicon) to ~100k surface forms while keeping the slang/emoji
    entries as the highest-priority source.
    """

    sentilex = {
        token: score * sentilex_weight
        for token, score in load_sentilex_flex().items()
    }
    lexicon = _merge_lexicons(
        sentilex,
        load_lexicon(source=OPLEXICON_LEXICON_SOURCE),
        load_twitter_extras(),
    )
    return SymbolicSentimentAnalyzer(
        lexicon=lexicon,
        tokenizer=tokenize_tweet,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        **kwargs,  # type: ignore[arg-type]
    )


@ANALYZER_REGISTRY.register("oplexicon_tweet_lemma")
def _build_oplexicon_tweet_lemma_analyzer(
    positive_threshold: float = 0.5,
    negative_threshold: float = -0.5,
    **kwargs: object,
) -> SymbolicSentimentAnalyzer:
    """``oplexicon_tweet`` variant that lemmatizes word tokens with spaCy.

    Same lexicon (OpLexicon + slang/emoji extras) and same default thresholds
    as :func:`_build_oplexicon_tweet_analyzer`, but uses
    :func:`tokenize_tweet_lemma`. Lemmatization recovers verb conjugations
    that are not explicitly listed in the slang dictionary
    ("curtindo"/"curtiu" -> "curtir") while still preserving emojis.
    """

    lexicon = _merge_lexicons(
        load_lexicon(source=OPLEXICON_LEXICON_SOURCE),
        load_twitter_extras(),
    )
    return SymbolicSentimentAnalyzer(
        lexicon=lexicon,
        tokenizer=tokenize_tweet_lemma,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        **kwargs,  # type: ignore[arg-type]
    )


class _MajorityAnalyzer:
    """Reference baseline that always predicts the same label."""

    def __init__(self, label: str = "positive") -> None:
        from pln_core.pipeline import AnalysisResult

        self._label = label
        self._result_cls = AnalysisResult

    def analyze(self, text: str):
        return self._result_cls(
            text=text,
            normalized_text=text,
            tokens=(),
            score=0.0,
            label=self._label,
            matched_terms=(),
        )


@ANALYZER_REGISTRY.register("majority")
def _build_majority_analyzer(label: str = "positive"):
    """Majority-class predictor (defaults to ``positive``, the largest class)."""

    return _MajorityAnalyzer(label=label)
