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

### desenvolvimento 20260612_113411_308889

Execucao intermediaria em CPU com `distilbert_multilingual`, 1000 exemplos por
classe no treino, 333 exemplos por classe no teste, 1 epoca, batch size 4 e
gradient accumulation 4. Esta execucao valida o comportamento do modelo em uma
amostra estratificada maior, mas ainda nao usa o teste comum completo.

| Modelo | Treino | Teste | Acuracia | Macro-F1 | F1 positivo | F1 negativo | F1 neutro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `distilbert_multilingual` dev | 3000 | 999 | 0.9970 | 0.9970 | 0.9955 | 0.9985 | 0.9970 |

Artefatos locais:

- resumo: `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_113411_308889/reports/summary_metrics.md`;
- modelo: `../../data/models/etapa2_subsymbolic/transformers/20260612_113411_308889/distilbert_multilingual/`.

### desenvolvimento 20260612_113749_208484

Execucao intermediaria em CPU com o mesmo treino estratificado de 3000 exemplos,
mas avaliada no teste comum completo de 4999 exemplos. Este e o primeiro
resultado neural de desenvolvimento diretamente comparavel ao teste usado pela
etapa 1 e pelos baselines TF-IDF, embora ainda nao seja o treino transformer
final no corpus completo.

| Modelo | Treino | Teste | Acuracia | Macro-F1 | F1 positivo | F1 negativo | F1 neutro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `distilbert_multilingual` dev/full-test | 3000 | 4999 | 0.9950 | 0.9950 | 0.9925 | 0.9979 | 0.9946 |

Artefatos locais:

- resumo: `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_113749_208484/reports/summary_metrics.md`;
- predicoes e erros: `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_113749_208484/`;
- modelo: `../../data/models/etapa2_subsymbolic/transformers/20260612_113749_208484/distilbert_multilingual/`.

Observacao para o relatorio: a pontuacao muito alta deve ser acompanhada por
uma checagem de duplicatas entre treino/teste e de artefatos lexicais dos
rotulos antes de ser interpretada como desempenho final do modelo.

Checagem inicial de duplicatas exatas normalizadas:

- treino: 99043 textos unicos em 100000 linhas;
- teste: 4996 textos unicos em 4999 linhas;
- interseccao treino/teste: 9 textos unicos, cobrindo 10 linhas do teste;
- conflitos de rotulo na interseccao: 0.

## analise de artefatos 20260612_115649_211081

Comando:

```bash
uv run python etapas/etapa2_subsimbolica/pipelines/run_data_artifact_analysis.py
```

Principais achados no teste comum:

- split quase balanceado: 1667 positivos, 1666 negativos e 1666 neutros;
- duplicacao exata treino/teste muito baixa: 9 textos normalizados em comum,
  cobrindo 10 linhas do teste, sem conflito de rotulo;
- 91,42% dos tweets positivos contem emoticon positivo;
- 99,88% dos tweets negativos contem emoticon negativo;
- 99,70% dos tweets neutros contem URL, sugerindo que a classe neutra e
  fortemente associada a noticias/links;
- os termos mais associados a neutro incluem fontes como `feedly`,
  `esportefera`, `estadaoeconomia`, `g1sp` e `cbn`.

Artefatos locais:

- resumo: `../../outputs/etapa2_subsymbolic/data_artifacts/20260612_115649_211081/reports/artifact_analysis.md`;
- tabelas: `../../outputs/etapa2_subsymbolic/data_artifacts/20260612_115649_211081/tables/`;
- figura: `../../outputs/etapa2_subsymbolic/data_artifacts/20260612_115649_211081/figures/cue_prevalence_test.png`.

Implicacao para o relatorio: resultados neurais muito altos devem ser
descritos junto com esses artefatos de supervisao distante, pois o modelo pode
aprender sinais superficiais do proprio protocolo de rotulagem.
