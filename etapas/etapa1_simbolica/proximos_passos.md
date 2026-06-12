# status e próximos passos da etapa 1

## concluído

- O corpus comum é o Kaggle "Portuguese Tweets for Sentiment Analysis".
- `TrainingDatasets/Train3Classes.csv` é usado para treino da etapa 2.
- `TestDatasets/Test3classes.csv` é o teste comum da etapa 1 e da etapa 2.
- O baseline oficial da etapa 1 é somente `oplexicon_regex`.
- A bateria simbólica roda o texto bruto e as duas condições tratadas usadas na etapa 2.
- Tabelas, predições, casos de erro e figuras são gerados em `../../outputs/etapa1_symbolic/benchmark_suite/<run_id>/`.

## relatório

- O relatório revisado da etapa 1 está em `reports/pln-core-part-1/`.
- A seção de resultados usa o corpus Kaggle comum.
- O texto deve tratar TweetSentBR apenas como contexto bibliográfico, não como corpus operacional.

## próximos passos restantes

- Selecionar exemplos reais de acerto/erro das predições salvas para enriquecer o relatório final.
- Manter a comparação com a etapa 2 sincronizada quando novos modelos forem integrados ao Streamlit.
- Não reintroduzir configs oficiais para variantes simbólicas antigas.

## comandos esperados

Depois de baixar o corpus Kaggle para `data/raw/portuguese-tweets-for-sentiment-analysis`, a bateria simbólica principal deve seguir este formato:

```bash
uv run python \
  etapas/etapa1_simbolica/pipelines/run_symbolic_benchmark_suite.py \
  'text_treatments=[raw,strip_emoticons_urls,strip_social_source_cues]'
```

Para rodar rapidamente no dataset didático:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py dataset=sample
```
