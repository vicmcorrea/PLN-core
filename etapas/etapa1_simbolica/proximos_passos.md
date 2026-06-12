# próximos passos da etapa 1

## corrigir dataset e avaliação

- Baixar o corpus Kaggle "Portuguese Tweets for Sentiment Analysis" localmente.
- Usar `TrainingDatasets/Train3Classes.csv` apenas como referência de treino para a etapa 2.
- Usar `TestDatasets/Test3classes.csv` como teste comum da etapa 1 e da etapa 2.
- Rodar os analisadores simbólicos atuais no teste multiclasse.
- Gerar novas tabelas com acurácia, macro-F1, F1 por classe e matriz de confusão.
- Separar exemplos de acerto e erro por classe para o relatório.

## corrigir texto do relatório

- Em "tokenização e lematização", mencionar apenas o método usado no fluxo real do app e da avaliação.
- Explicar a solução como uma arquitetura única: entrada de texto, normalização, tokenização, consulta aos léxicos, regras simbólicas, cálculo de polaridade, saída e visualização.
- Evitar apresentar Streamlit, avaliação e núcleo simbólico como entidades separadas; eles são partes de um fluxo único.
- Seção de corpus atualizada para remover dependência operacional de TweetSentBR.
- Atualizar a seção de resultados com o corpus comum escolhido para as duas etapas.

## reorganizar estrutura

- Estrutura base separada em `configs/`, `pipelines/`, `experiments/` e `reports/`.
- Manter Streamlit e avaliação funcionando a partir do mesmo núcleo `pln_core`.
- CLI e entry points relacionados removidos do fluxo ativo.
- Manter o dataset `sample` como teste rápido, mas não usá-lo como resultado principal.
- Garantir que os recursos léxicos necessários sejam documentados e carregáveis no ambiente dos professores.

## comandos esperados

Depois de baixar o corpus Kaggle para `data/raw/portuguese-tweets-for-sentiment-analysis`, a avaliação simbólica deve seguir este formato:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py
```

Para rodar rapidamente no dataset didático:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py dataset=sample
```
