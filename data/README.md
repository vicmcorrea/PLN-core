# data

Dados e recursos compartilhados pelas duas etapas.

## corpus comum

Use o dataset Kaggle `augustop/portuguese-tweets-for-sentiment-analysis` para as duas etapas. A estrutura local esperada e ignorada pelo Git e:

```text
data/raw/portuguese-tweets-for-sentiment-analysis/
  TrainingDatasets/Train3Classes.csv
  TestDatasets/Test3classes.csv
```

`Train3Classes.csv` deve ser usado para treinar modelos estatisticos/neuronais da etapa 2. `Test3classes.csv` deve ser o teste comum para comparar a etapa 1 simbolica com a etapa 2.

Além do texto bruto (`raw`), os pipelines compartilham tratamentos de texto
para diagnosticar vazamento de rótulo:

- `strip_emoticons_urls`: remove emoticons e URLs.
- `strip_social_source_cues`: remove emoticons, URLs, menções, hashtags e
  marcadores recorrentes de fonte.

## recursos versionados

- `recommendations.json`: recomendacoes usadas pelo Streamlit.
- `../src/pln_core/data/seed_lexicon.json`: lexico didatico pequeno.
- `../src/pln_core/data/slang_emoji_ptbr.tsv`: extensao historica curta de
  girias e emojis; nao entra no baseline oficial `oplexicon_regex`.

## recursos locais

Arquivos grandes e recursos externos completos, como OpLexicon quando baixado
localmente, ficam em subpastas ignoradas pelo Git.

Modelos treinados também são locais e ignorados pelo Git. A aplicação
Streamlit procura artefatos TF-IDF em:

```text
data/models/etapa2_subsymbolic/<run_id>/<model>.joblib
data/models/etapa2_subsymbolic/<run_id>/<model>.metadata.json
```

Para o estado atual do projeto, o modelo padrão esperado é
`tfidf_logreg` treinado com `strip_emoticons_urls`. A execução local de
referência é `20260614_113447_389024`.
