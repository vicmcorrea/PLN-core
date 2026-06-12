# etapa 1 simbólica

Esta pasta organiza a primeira etapa depois das correções pedidas na revisão do professor. A implementação simbólica continua em `../../src/pln_core`, porque o Streamlit, os testes e o harness de avaliação compartilham o mesmo núcleo.

## escopo

A etapa 1 contém a solução simbólica de análise de sentimentos em português brasileiro:
- normalização textual;
- tokenização usada pelo fluxo real do app e da avaliação;
- consulta ao OpLexicon v3.0;
- tratamento de negação, intensificação e sinais simples de linguagem de redes sociais;
- classificação em positivo, negativo ou neutro;
- execução principal via Streamlit;
- avaliação por acurácia, macro-F1, F1 por classe e matriz de confusão.

## estrutura

- `configs/`: configuração Hydra da avaliação simbólica.
- `pipelines/`: entrada executável da etapa 1.
- `experiments/`: índice dos experimentos simbólicos e saídas esperadas.
- `reports/`: espaço do relatório e LaTeX desta etapa.
- `proximos_passos.md`: checklist específico da etapa 1.

## corpus comum

A versão revisada da etapa 1 deve ser avaliada no mesmo corpus usado pela etapa 2. O corpus operacional passa a ser o Kaggle "Portuguese Tweets for Sentiment Analysis", usando `TestDatasets/Test3classes.csv` como teste comum de comparação.

Não usar TweetSentBR como corpus operacional desta etapa. A justificativa é reprodutibilidade: a distribuição original depende de ids de tweets e hidratação via API, enquanto o corpus Kaggle pode ser replicado diretamente via Kaggle.

## comandos

Rodar o baseline oficial no corpus comum:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_benchmark_suite.py
```

Rodar uma avaliação individual com a configuração principal:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py
```

Rodar o smoke test didático:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py dataset=sample
```

Os relatórios da etapa 1 são gerados em `../../outputs/etapa1_symbolic/benchmark_suite/<run_id>/` para a bateria oficial e em `../../outputs/etapa1_symbolic/runs/<run_id>/<dataset>/<analyzer>/` para execuções individuais. O `run_id` é criado com timestamp e microsegundos para evitar sobrescrita entre execuções.

## arquivos relacionados

- `../../src/pln_core/pipeline.py`: regras simbólicas e agregação de polaridade.
- `../../src/pln_core/eval/`: harness compartilhado de avaliação.
- `../../src/pln_core/eval/datasets/kaggle_tweets.py`: loader do corpus comum.
- `../../streamlit_app.py`: demonstração interativa da solução simbólica.
