from __future__ import annotations

from pln_core.lexicon import OPLEXICON_LEXICON_SOURCE, SEED_LEXICON_SOURCE
from pln_core.tokenizers import CUSTOM_TOKENIZER_SOURCE, SPACY_PT_TOKENIZER_SOURCE

ANALYZE_MODE = "analyze"
COMPARE_MODE = "compare"

SAMPLE_TEXTS: dict[str, str] = {
    "positive": "Eu amei o filme, foi muito bom!",
    "negative": "Nao gostei do app, esta bem confuso e bugado.",
    "neutral": "Hoje eu fui ao cinema e voltei para casa depois da sessao.",
}

COMPARISON_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("positive", SAMPLE_TEXTS["positive"]),
    ("negative", SAMPLE_TEXTS["negative"]),
    ("neutral", SAMPLE_TEXTS["neutral"]),
    ("social", "@ana eu muuuuito gostei disso :) #cinema"),
)

LEXICON_SOURCE_LABELS: dict[str, str] = {
    SEED_LEXICON_SOURCE: "use the built-in seed dictionary",
    OPLEXICON_LEXICON_SOURCE: "use oplexicon v3.0",
}

LEXICON_SOURCE_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("1", SEED_LEXICON_SOURCE, LEXICON_SOURCE_LABELS[SEED_LEXICON_SOURCE]),
    ("2", OPLEXICON_LEXICON_SOURCE, LEXICON_SOURCE_LABELS[OPLEXICON_LEXICON_SOURCE]),
)

TOKENIZER_SOURCE_LABELS: dict[str, str] = {
    CUSTOM_TOKENIZER_SOURCE: "use the built-in regex tokenizer",
    SPACY_PT_TOKENIZER_SOURCE: "use spaCy portuguese tokenizer",
}

TOKENIZER_SOURCE_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("1", CUSTOM_TOKENIZER_SOURCE, TOKENIZER_SOURCE_LABELS[CUSTOM_TOKENIZER_SOURCE]),
    (
        "2",
        SPACY_PT_TOKENIZER_SOURCE,
        TOKENIZER_SOURCE_LABELS[SPACY_PT_TOKENIZER_SOURCE],
    ),
)

START_MODE_LABELS: dict[str, str] = {
    ANALYZE_MODE: "analyze one text",
    COMPARE_MODE: "run comparison examples",
}

START_MODE_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("1", ANALYZE_MODE, START_MODE_LABELS[ANALYZE_MODE]),
    ("2", COMPARE_MODE, START_MODE_LABELS[COMPARE_MODE]),
)

MENU_OPTIONS: tuple[tuple[str, str], ...] = (
    ("1", "Write your own text"),
    ("2", "Use a positive sample"),
    ("3", "Use a negative sample"),
    ("4", "Use a neutral sample"),
)
