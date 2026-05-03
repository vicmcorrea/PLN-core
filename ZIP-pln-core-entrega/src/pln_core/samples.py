from __future__ import annotations

ANALYZE_MODE = "analyze"
COMPARE_MODE = "compare"
ANALYZER_STACK_LABEL = "OpLexicon v3.0 + spaCy Portuguese lemmatizer"

SAMPLE_TEXTS: dict[str, str] = {
    "positive": "Eu amei o filme, foi muito bom!",
    "negative": "Nao gostei do app, esta bem confuso e bugado.",
    "neutral": "O arquivo tem quatro paginas e duas tabelas.",
}

COMPARISON_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("positive", SAMPLE_TEXTS["positive"]),
    ("negative", SAMPLE_TEXTS["negative"]),
    ("neutral", SAMPLE_TEXTS["neutral"]),
    ("social", "@ana eu muuuuito gostei disso :) #cinema"),
)

START_MODE_LABELS: dict[str, str] = {
    ANALYZE_MODE: "analyze one text",
    COMPARE_MODE: "run the built-in example set",
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
