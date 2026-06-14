# pipelines da etapa 2

Esta pasta guarda os scripts de treino e avaliação subsimbólica.

Os scripts daqui devem depender de código reutilizável em `src/pln_core`, mas não devem misturar resultados ou configurações com a etapa 1.

## suite clássica TF-IDF

O pipeline `run_classical_benchmark_suite.py` treina e avalia os baselines
classicos no corpus Kaggle comum:

```bash
uv run python etapas/etapa2_subsimbolica/pipelines/run_classical_benchmark_suite.py
```

Modelos executados por padrão:

- `tfidf_logreg`: TF-IDF + Regressao Logistica.
- `tfidf_linear_svm`: TF-IDF + Linear SVM.

O parametro Hydra `text_treatment` controla o texto entregue ao modelo. O
default `raw` preserva o split Kaggle original. Para experimentos de robustez,
use `text_treatment=strip_emoticons_urls`. Para um diagnostico exploratorio
mais forte contra pistas sociais e fontes de noticias, use
`text_treatment=strip_social_source_cues`.

Entradas:

- treino: `../../data/raw/portuguese-tweets-for-sentiment-analysis/TrainingDatasets/Train3Classes.csv`;
- teste comum: `../../data/raw/portuguese-tweets-for-sentiment-analysis/TestDatasets/Test3classes.csv`;
- config principal: `../configs/benchmark_suite.yaml`;
- configs de modelo: `../configs/model/`.

Saidas por execucao:

- relatorios e tabelas: `../../outputs/etapa2_subsymbolic/benchmark_suite/<run_id>/reports/`;
- predicoes: `../../outputs/etapa2_subsymbolic/benchmark_suite/<run_id>/predictions/`;
- casos de erro: `../../outputs/etapa2_subsymbolic/benchmark_suite/<run_id>/cases/`;
- figuras: `../../outputs/etapa2_subsymbolic/benchmark_suite/<run_id>/figures/`;
- modelos exportados: `../../data/models/etapa2_subsymbolic/<run_id>/`.

## analise de artefatos do corpus

O pipeline `run_data_artifact_analysis.py` mede duplicatas exatas e pistas
lexicais simples do corpus Kaggle comum. Ele ajuda a interpretar resultados
muito altos em modelos neurais, ja que o dataset foi rotulado por supervisao
distante com sinais de tweets.

```bash
uv run python etapas/etapa2_subsimbolica/pipelines/run_data_artifact_analysis.py
```

Entradas:

- treino e teste definidos em `../configs/dataset/kaggle_portuguese_tweets.yaml`;
- config principal: `../configs/data_artifact_analysis.yaml`.

Saidas por execucao:

- resumo: `../../outputs/etapa2_subsymbolic/data_artifacts/<run_id>/reports/artifact_analysis.md`;
- tabelas: `../../outputs/etapa2_subsymbolic/data_artifacts/<run_id>/tables/`;
- figuras: `../../outputs/etapa2_subsymbolic/data_artifacts/<run_id>/figures/`;
- metadados Hydra: `../../outputs/etapa2_subsymbolic/data_artifacts/_hydra/<run_id>/`.

## diagnostico de vazamento por pistas superficiais

O pipeline `run_leakage_diagnostics.py` testa se emoticons, URLs e outras
pistas superficiais explicam os resultados muito altos no split Kaggle. Ele
roda:

- regra simples baseada em emoticon negativo, emoticon positivo e URL;
- Regressao Logistica usando apenas `has_positive_emoticon`,
  `has_negative_emoticon` e `has_url`;
- TF-IDF + Regressao Logistica e TF-IDF + Linear SVM em texto bruto e em texto
  tratado. O tratamento padrao do diagnostico remove emoticons/URLs; o
  tratamento exploratorio `strip_social_source_cues` tambem remove mencoes,
  hashtags e marcadores de fontes neutras identificados na analise de artefatos.

```bash
uv run python etapas/etapa2_subsimbolica/pipelines/run_leakage_diagnostics.py
```

Diagnostico exploratorio mais forte:

```bash
uv run python etapas/etapa2_subsimbolica/pipelines/run_leakage_diagnostics.py stripped_treatment=strip_social_source_cues
```

Saidas por execucao:

- resumo: `../../outputs/etapa2_subsymbolic/leakage_diagnostics/<run_id>/reports/summary_metrics.md`;
- tabela de pistas: `../../outputs/etapa2_subsymbolic/leakage_diagnostics/<run_id>/tables/cue_prevalence.csv`;
- predicoes e erros: `../../outputs/etapa2_subsymbolic/leakage_diagnostics/<run_id>/`;
- figuras: `../../outputs/etapa2_subsymbolic/leakage_diagnostics/<run_id>/figures/`.

## fine-tuning transformer

O pipeline `run_transformer_benchmark.py` usa Hugging Face Transformers para
treinar um modelo de classificacao de sequencias com tres rotulos
(`positive`, `negative`, `neutral`).

Instalacao opcional:

```bash
uv sync --extra transformers
```

Smoke test recomendado antes da execucao completa:

```bash
uv run --extra transformers python etapas/etapa2_subsimbolica/pipelines/run_transformer_benchmark.py model=distilbert_multilingual train_max_examples=120 test_max_examples=60 model.training.epochs=1 trainer.use_cpu=true
```

Execucao final planejada com XLM-R base:

```bash
uv run --extra transformers python etapas/etapa2_subsimbolica/pipelines/run_transformer_benchmark.py model=xlm_roberta_base
```

Para remover emoticons e URLs tanto no treino quanto no teste:

```bash
uv run --extra transformers python etapas/etapa2_subsimbolica/pipelines/run_transformer_benchmark.py model=distilbert_multilingual train_per_class=1000 model.training.epochs=1 trainer.use_cpu=true text_treatment=strip_emoticons_urls
```

Execucoes de desenvolvimento em CPU ja usadas para comparar arquiteturas no
teste comum completo:

```bash
uv run --extra transformers python etapas/etapa2_subsimbolica/pipelines/run_transformer_benchmark.py model=xlm_roberta_base train_per_class=1000 model.training.epochs=1 trainer.use_cpu=true
uv run --extra transformers python etapas/etapa2_subsimbolica/pipelines/run_transformer_benchmark.py model=albertina_ptbr_100m train_per_class=1000 model.training.epochs=1 trainer.use_cpu=true
```

Em ambiente sem GPU dedicada, rode primeiro com `trainer.use_cpu=true` e uma
amostra pequena. Em Mac com MPS, os defaults usam lotes menores, mas o treino
completo ainda pode exigir reduzir `model.training.batch_size` ou usar CPU.

Saidas por execucao:

- relatorios: `../../outputs/etapa2_subsymbolic/transformer_benchmark/<run_id>/reports/`;
- predicoes e erros: `../../outputs/etapa2_subsymbolic/transformer_benchmark/<run_id>/`;
- figuras: `../../outputs/etapa2_subsymbolic/transformer_benchmark/<run_id>/figures/`;
- checkpoints/modelo final: `../../data/models/etapa2_subsymbolic/transformers/<run_id>/<model>/`.

## ordem de evolução

1. preparação do corpus Kaggle;
2. treino TF-IDF + Regressão Logística;
3. treino TF-IDF + Linear SVM;
4. fine-tuning de transformer;
5. consolidação dos resultados para comparação com a etapa 1.
