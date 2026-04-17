from __future__ import annotations

from pln_core.lexicon import OPLEXICON_LEXICON_SOURCE, SEED_LEXICON_SOURCE

SAMPLE_TEXTS: dict[str, str] = {
    "positive": "Eu amei o filme, foi muito bom!",
    "negative": "Nao gostei do app, esta bem confuso e bugado.",
    "neutral": "Hoje eu fui ao cinema e voltei para casa depois da sessao.",
}

LEXICON_SOURCE_LABELS: dict[str, str] = {
    SEED_LEXICON_SOURCE: "use the built-in seed dictionary",
    OPLEXICON_LEXICON_SOURCE: "use oplexicon v3.0",
}

LEXICON_SOURCE_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("1", SEED_LEXICON_SOURCE, LEXICON_SOURCE_LABELS[SEED_LEXICON_SOURCE]),
    ("2", OPLEXICON_LEXICON_SOURCE, LEXICON_SOURCE_LABELS[OPLEXICON_LEXICON_SOURCE]),
)

MENU_OPTIONS: tuple[tuple[str, str], ...] = (
    ("1", "Write your own text"),
    ("2", "Use a positive sample"),
    ("3", "Use a negative sample"),
    ("4", "Use a neutral sample"),
)
