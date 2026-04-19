# pln-core

symbolic sentiment analysis for brazilian portuguese.

stack: oplexicon v3.0 + spacy pt tokenizer, with rules for negation, intensifiers, diminishers, contrast and exclamation.

## setup

```bash
./install.sh
```

uses `uv` if available, otherwise creates `.venv/`.

## cli

```bash
source .venv/bin/activate
python main.py
```

or direct:

```bash
python main.py "eu amei o filme, foi muito bom!"
python main.py --sample positive
python main.py --compare
python main.py "nao gostei do app." --json
```

## streamlit app

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

opens at `http://localhost:8501`. pick a sample or paste text, hit `analisar`, and get the label, score, tokens, matched terms, and one song suggestion matching the sentiment.

## music recommendations

catalog at `data/recommendations.json`: 10 brazilian songs with `sentiment` and `valence` in `[-1, 1]`. the recommender filters by label and ranks by valence proximity to the analyzer score. add entries directly in the json.

## tests

```bash
source .venv/bin/activate
python -m unittest discover -s tests
```

## notes

- oplexicon is downloaded on first use and cached under `data/external/`.
