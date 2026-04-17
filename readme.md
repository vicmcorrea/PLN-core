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

1. choose the cli mode
2. if analysis mode, choose the lexicon source
3. choose the tokenizer source
4. choose the text input

cli modes:

1. analyze one text
2. run comparison examples

lexicon sources:

1. built-in seed dictionary
2. oplexicon v3.0

tokenizer sources:

1. built-in regex tokenizer
2. spaCy portuguese tokenizer

comparison mode runs built-in examples across:

- seed dictionary + regex tokenizer
- seed dictionary + spaCy portuguese tokenizer
- oplexicon v3.0 + regex tokenizer
- oplexicon v3.0 + spaCy portuguese tokenizer

analysis mode prints the result and asks if you want to analyze another text.

## tests

```bash
uv run python -m unittest discover -s tests
```
