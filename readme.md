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

## direct use

```bash
uv run python main.py --sample positive
uv run python main.py --sample negative
uv run python main.py --sample neutral
uv run python main.py "nao gostei do app"
```

## tests

```bash
uv run python -m unittest discover -s tests
```
