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

1. write your own text
2. use a positive sample
3. use a negative sample
4. use a neutral sample

then it prints the analysis and asks if you want to analyze another text.

## tests

```bash
uv run python -m unittest discover -s tests
```
