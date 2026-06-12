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
  transformer_benchmark/<run_id>/
    reports/
      resolved_config.json
      dataset_manifest.json
      summary_metrics.md
      <model>/report.json
    predictions/<model>.csv
    cases/<model>_errors.csv
    figures/
      confusion_<model>.png
      confusion_<model>.pdf
  data_artifacts/<run_id>/
    reports/
      artifact_analysis.json
      artifact_analysis.md
    tables/
      cue_prevalence.csv
      label_associated_terms.csv
    figures/
      cue_prevalence_test.png
      cue_prevalence_test.pdf
  leakage_diagnostics/<run_id>/
    reports/
      resolved_config.json
      dataset_manifest.json
      leakage_diagnostics.json
      summary_metrics.csv
      summary_metrics.md
    tables/
      cue_prevalence.csv
    predictions/<diagnostic>.csv
    cases/<diagnostic>_errors.csv
    figures/
      diagnostic_accuracy.png
      diagnostic_accuracy.pdf
      diagnostic_macro_f1.png
      diagnostic_macro_f1.pdf
      confusion_<cue_diagnostic>.png
      confusion_<cue_diagnostic>.pdf
  multirun/<run_id>/<model>/

data/models/etapa2_subsymbolic/
  <run_id>/<model>.joblib
  transformers/<run_id>/<model>/
    Hugging Face model checkpoint
```

Treinos TF-IDF devem salvar metricas, predicoes e, se necessario, um artefato `.joblib` em `data/models/etapa2_subsymbolic/<run_id>/`. Treinos transformer devem usar `transformer_benchmark/<run_id>/` para resultados e `data/models/etapa2_subsymbolic/transformers/<run_id>/<model>/` para checkpoints/modelo final.
Analises de artefatos do corpus devem usar `data_artifacts/<run_id>/` e nunca
misturar tabelas de dataset com relatorios de modelos.
Diagnosticos de vazamento por pistas superficiais devem usar
`leakage_diagnostics/<run_id>/`, separado tanto de `benchmark_suite/` quanto de
`transformer_benchmark/`.

## regra pratica

Nunca salve diretamente em `outputs/<etapa>/<model>/report.json`. Sempre inclua uma categoria de pipeline e um `run_id`, como `benchmark_suite/<run_id>/`, para preservar execucoes anteriores.
