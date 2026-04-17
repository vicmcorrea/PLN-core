# pln-core

symbolic sentiment analysis baseline for portuguese.

## setup

```bash
uv sync
```

## demo

```bash
uv run python main.py
```

after running it, the terminal app opens and shows:

1. choose the lexicon source
2. choose the text input

lexicon sources:

1. built-in seed dictionary
2. oplexicon v3.0

then it prints the analysis and asks if you want to analyze another text.

## tests

```bash
uv run python -m unittest discover -s tests
```
