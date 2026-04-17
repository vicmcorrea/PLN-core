from __future__ import annotations

SAMPLE_TEXTS: dict[str, str] = {
    "positive": "Eu amei o filme, foi muito bom!",
    "negative": "Nao gostei do app, esta bem confuso e bugado.",
    "neutral": "Hoje eu fui ao cinema e voltei para casa depois da sessao.",
}

MENU_OPTIONS: tuple[tuple[str, str], ...] = (
    ("1", "Write your own text"),
    ("2", "Use a positive sample"),
    ("3", "Use a negative sample"),
    ("4", "Use a neutral sample"),
)
