# experimentos da etapa 1

Esta pasta registra os experimentos simbolicos que precisam aparecer no relatorio da etapa 1.

Resultados gerados automaticamente devem ficar em `../../outputs/etapa1_symbolic/runs/<run_id>/` ou `../../outputs/etapa1_symbolic/benchmark_suite/<run_id>/`, que nao entram no Git. Quando um resultado for aprovado para o relatorio, registre aqui a configuracao usada, o caminho do JSON e um resumo curto das metricas.

Experimentos minimos consolidados:

1. `oplexicon_regex` no teste Kaggle multiclasse, como baseline oficial da Etapa 1.
2. `oplexicon_regex` nas condicoes `strip_emoticons_urls` e `strip_social_source_cues`, para comparacao justa com a Etapa 2.
3. `sample` apenas como smoke test de regressao.

## execucao consolidada 20260612_152415_135433

Comando:

```bash
uv run python \
  etapas/etapa1_simbolica/pipelines/run_symbolic_benchmark_suite.py \
  'text_treatments=[raw,strip_emoticons_urls,strip_social_source_cues]'
```

| Analisador | Tratamento | Acuracia | Macro-F1 | F1 positivo | F1 negativo | F1 neutro |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `oplexicon_regex` | `raw` | 0.5979 | 0.5960 | 0.6515 | 0.6407 | 0.4956 |
| `oplexicon_regex` | `strip_emoticons_urls` | 0.3697 | 0.3668 | 0.3959 | 0.3174 | 0.3870 |
| `oplexicon_regex` | `strip_social_source_cues` | 0.3695 | 0.3665 | 0.3959 | 0.3169 | 0.3869 |

Artefatos locais:

- resumo: `../../outputs/etapa1_symbolic/benchmark_suite/20260612_152415_135433/reports/summary_metrics.md`;
- predicoes e erros: `../../outputs/etapa1_symbolic/benchmark_suite/20260612_152415_135433/`;
- figuras: `../../outputs/etapa1_symbolic/benchmark_suite/20260612_152415_135433/figures/`.
