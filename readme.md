# pln-core

Symbolic sentiment analysis baseline for Brazilian Portuguese.

## Current stack

The project now runs with one fixed configuration:

- `OpLexicon v3.0` for lexical polarity
- spaCy Portuguese tokenizer for tokenization

The symbolic scoring rules remain in the project code: negation, intensifiers, diminishers, contrast, and exclamation marks.

## Setup

Use the install script from the project root:

```bash
./install.sh
```

If `uv` is available, the script uses it. Otherwise it creates `.venv/` and installs the package there.

## Run the CLI

If you have `uv`:

```bash
uv run python main.py
```

If you are using the virtual environment created by `install.sh`:

```bash
source .venv/bin/activate
python main.py
```

## CLI flow

After opening the CLI, the analyzer always uses `OpLexicon v3.0 + spaCy Portuguese tokenizer`.

You can choose between:

1. `analyze one text`
2. `run the built-in example set`

In analyze mode, you can:

1. write your own text
2. use a positive sample
3. use a negative sample
4. use a neutral sample

The CLI prints the normalized text, tokens, final label, score, and matched terms with the symbolic rules applied.

## Direct usage

Analyze one text directly:

```bash
uv run python main.py "Eu amei o filme, foi muito bom!"
```

Use a built-in sample directly:

```bash
uv run python main.py --sample positive
```

Run the built-in example set:

```bash
uv run python main.py --compare
```

Print JSON output:

```bash
uv run python main.py "Nao gostei do app." --json
```

## Run the Streamlit app

The same analyzer is also available as a small web app.

With `uv`:

```bash
uv run streamlit run streamlit_app.py
```

With `.venv`:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501` on a single screen. You can:

1. Pick a built-in sample (`positive`, `negative`, `neutral`) or paste your own text.
2. Click `Analyze` to run the analyzer.
3. Inspect the label, score, normalized text, tokens, and the matched terms with the symbolic rules applied.
4. Click `Clear` to reset the text and the result.

## Notes

- On the first use of `OpLexicon`, the project downloads `lexico_v3.0.txt` and caches it under `data/external/`.
- If the download fails, check your internet connection and run the command again.

## Tests

With `uv`:

```bash
uv run python -m unittest discover -s tests
```

With `.venv`:

```bash
source .venv/bin/activate
python -m unittest discover -s tests
```
