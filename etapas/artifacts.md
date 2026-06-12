# organizacao de outputs e artefatos

Este projeto separa resultados gerados por etapa e por execucao. Nada dentro de `outputs/`, `data/raw/`, `data/processed/` ou `data/models/` deve ser versionado.

## etapa 1

```text
outputs/etapa1_symbolic/
  runs/<run_id>/<dataset>/<analyzer>/
    report.json
  benchmark_suite/<run_id>/
    dataset/
      dataset_profile.json
      train_normalized.csv
      test_normalized.csv
    reports/
      summary_metrics.csv
      summary_metrics.md
      run_manifest.json
    predictions/<analyzer>.csv
    cases/<analyzer>_errors.csv
    figures/
  multirun/<run_id>/<analyzer>/
```

O `run_id` vem da configuracao Hydra em `etapa1_simbolica/configs/default.yaml` e inclui timestamp com microsegundos.

## etapa 2

```text
outputs/etapa2_subsymbolic/
  benchmark_suite/<run_id>/
    reports/
      resolved_config.json
      dataset_manifest.json
      summary_metrics.csv
      summary_metrics.md
      <model>/report.json
    predictions/<model>.csv
    cases/<model>_errors.csv
    figures/
      confusion_<model>.png
      confusion_<model>.pdf
      benchmark_accuracy.png
      benchmark_accuracy.pdf
      benchmark_macro_f1.png
      benchmark_macro_f1.pdf
    errors/<model>.txt
  multirun/<run_id>/<model>/

data/models/etapa2_subsymbolic/
  <run_id>/<model>.joblib
```

Treinos TF-IDF devem salvar metricas, predicoes e, se necessario, um artefato `.joblib` em `data/models/etapa2_subsymbolic/<run_id>/`. Treinos transformer devem usar uma subpasta propria sob `outputs/etapa2_subsymbolic/` e salvar checkpoints apenas quando forem necessarios para reproducao, preferencialmente com uma nota no experimento indicando modelo base, seed e hiperparametros.

## regra pratica

Nunca salve diretamente em `outputs/<etapa>/<model>/report.json`. Sempre inclua uma categoria de pipeline e um `run_id`, como `benchmark_suite/<run_id>/`, para preservar execucoes anteriores.
