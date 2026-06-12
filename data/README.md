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

## recursos versionados

- `recommendations.json`: recomendacoes usadas pelo Streamlit.
- `../src/pln_core/data/seed_lexicon.json`: lexico didatico pequeno.
- `../src/pln_core/data/slang_emoji_ptbr.tsv`: extensao curta de girias e emojis.

## recursos locais

Arquivos grandes e recursos externos completos, como OpLexicon e SentiLex quando baixados localmente, ficam em subpastas ignoradas pelo Git.
