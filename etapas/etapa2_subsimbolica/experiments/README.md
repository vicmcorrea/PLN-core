# experimentos da etapa 2

Esta pasta registra os experimentos subsimbolicos. Artefatos grandes devem ficar em `../../outputs/etapa2_subsymbolic/benchmark_suite/<run_id>/` ou `../../data/models/etapa2_subsymbolic/<run_id>/`, ambos ignorados pelo Git.

Experimentos minimos:

1. TF-IDF + Regressao Logistica.
2. TF-IDF + Linear SVM.
3. Fine-tuning de `FacebookAI/xlm-roberta-base` ou alternativa viavel.
4. Avaliacao opcional de `PORTULAN/albertina-100m-portuguese-ptbr-encoder`.
5. Tabela comparativa final contra a etapa 1 simbolica.

Todo experimento deve registrar dataset, split, pre-processamento, seed, hiperparametros principais, metricas e caminho do relatorio JSON.

## suite classica inicial

Comando:

```bash
uv run python etapas/etapa2_subsimbolica/pipelines/run_classical_benchmark_suite.py
```

Modelos:

- `tfidf_logreg`: TF-IDF word n-grams `(1, 2)`, `min_df=2`, `max_df=0.95`, `sublinear_tf=true`, `LogisticRegression(max_iter=1000, class_weight=balanced, random_state=42)`.
- `tfidf_linear_svm`: mesma vetorizacao, `LinearSVC(class_weight=balanced, random_state=42)`.

Baseline simbolico usado para comparacao preliminar:

- `oplexicon_regex`, etapa 1, run `20260612_101202_400841`: acuracia `0.5979`, macro-F1 `0.5960`.

Registrar abaixo o `run_id` e as metricas obtidas a cada execucao consolidada.

## execucoes consolidadas

### 20260612_103137_624831

Corpus comum Kaggle, treino `Train3Classes.csv` com 100000 exemplos e teste
`Test3classes.csv` com 4999 exemplos. Ambos os modelos usaram a mesma
vetorizacao TF-IDF word n-grams `(1, 2)`, `min_df=2`, `max_df=0.95` e
`sublinear_tf=true`.

| Modelo | Acuracia | Macro-F1 | F1 positivo | F1 negativo | F1 neutro | Vocabulario |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tfidf_logreg` | 0.8172 | 0.8164 | 0.7374 | 0.7421 | 0.9697 | 159164 |
| `tfidf_linear_svm` | 0.8080 | 0.8084 | 0.7223 | 0.7351 | 0.9677 | 159164 |

Artefatos locais:

- resumo: `../../outputs/etapa2_subsymbolic/benchmark_suite/20260612_103137_624831/reports/summary_metrics.md`;
- predicoes e erros: `../../outputs/etapa2_subsymbolic/benchmark_suite/20260612_103137_624831/`;
- modelos: `../../data/models/etapa2_subsymbolic/20260612_103137_624831/`.

## transformer pendente

Pipeline criado: `../pipelines/run_transformer_benchmark.py`.

Smoke test sugerido:

```bash
uv run --extra transformers python etapas/etapa2_subsimbolica/pipelines/run_transformer_benchmark.py model=distilbert_multilingual train_max_examples=120 test_max_examples=60 model.training.epochs=1 trainer.use_cpu=true
```

Execucao final planejada:

```bash
uv run --extra transformers python etapas/etapa2_subsimbolica/pipelines/run_transformer_benchmark.py model=xlm_roberta_base
```

Registrar aqui o `run_id`, metricas, tempo, hardware usado e se o treino usou
o split completo ou uma amostra estratificada.

### smoke test 20260612_113232_434909

Execucao de validacao do pipeline, nao comparavel como resultado final:
`distilbert_multilingual`, CPU, 120 exemplos de treino, 60 exemplos de teste,
1 epoca, batch size 4, gradient accumulation 4.

| Modelo | Acuracia | Macro-F1 | F1 positivo | F1 negativo | F1 neutro |
| --- | ---: | ---: | ---: | ---: | ---: |
| `distilbert_multilingual` smoke | 0.4667 | 0.2510 | 0.6279 | 0.1250 | 0.0000 |

Artefatos locais:

- resumo: `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_113232_434909/reports/summary_metrics.md`;
- predicoes e erros: `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_113232_434909/`;
- modelo: `../../data/models/etapa2_subsymbolic/transformers/20260612_113232_434909/distilbert_multilingual/`.

Observacao operacional: tentativas anteriores em MPS com 3000/999 exemplos e
600/300 exemplos falharam por falta de memoria compartilhada. O pipeline agora
possui `trainer.use_cpu=true` para smoke tests e configura lotes menores nos
modelos transformer.
