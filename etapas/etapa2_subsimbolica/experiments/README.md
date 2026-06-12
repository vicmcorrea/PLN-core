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

## transformer desenvolvimento

Pipeline criado: `../pipelines/run_transformer_benchmark.py`.

Smoke test sugerido:

```bash
uv run --extra transformers python etapas/etapa2_subsimbolica/pipelines/run_transformer_benchmark.py model=distilbert_multilingual train_max_examples=120 test_max_examples=60 model.training.epochs=1 trainer.use_cpu=true
```

Execucao de desenvolvimento no teste completo, usando 1000 exemplos por classe
no treino:

```bash
uv run --extra transformers python etapas/etapa2_subsimbolica/pipelines/run_transformer_benchmark.py model=xlm_roberta_base train_per_class=1000 model.training.epochs=1 trainer.use_cpu=true
uv run --extra transformers python etapas/etapa2_subsimbolica/pipelines/run_transformer_benchmark.py model=albertina_ptbr_100m train_per_class=1000 model.training.epochs=1 trainer.use_cpu=true
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

### desenvolvimento 20260612_115913_159120

Execucao em CPU com `FacebookAI/xlm-roberta-base`, 1000 exemplos por classe no
treino, teste comum completo, 1 epoca, batch size 2 e gradient accumulation 8.
O console registrou `train_runtime=853.4322s` e `eval_runtime=102.1262s`.

| Modelo | Treino | Teste | Acuracia | Macro-F1 | F1 positivo | F1 negativo | F1 neutro | Erros |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `xlm_roberta_base` dev/full-test | 3000 | 4999 | 0.9968 | 0.9968 | 0.9952 | 0.9985 | 0.9967 | 16 |

Confusao principal:

- positivo -> negativo: 5; positivo -> neutro: 1;
- negativo -> positivo/neutro: 0;
- neutro -> positivo: 10.

Artefatos locais:

- resumo: `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_115913_159120/reports/summary_metrics.md`;
- relatorio JSON: `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_115913_159120/reports/xlm_roberta_base/report.json`;
- predicoes e erros: `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_115913_159120/`;
- modelo: `../../data/models/etapa2_subsymbolic/transformers/20260612_115913_159120/xlm_roberta_base/`.

### desenvolvimento 20260612_121606_196690

Execucao em CPU com
`PORTULAN/albertina-100m-portuguese-ptbr-encoder`, 1000 exemplos por classe no
treino, teste comum completo, 1 epoca, batch size 4 e gradient accumulation 4.
O console registrou `train_runtime=711.2023s` e `eval_runtime=173.2008s`.

| Modelo | Treino | Teste | Acuracia | Macro-F1 | F1 positivo | F1 negativo | F1 neutro | Erros |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `albertina_ptbr_100m` dev/full-test | 3000 | 4999 | 0.9972 | 0.9972 | 0.9958 | 0.9973 | 0.9985 | 14 |

Confusao principal:

- positivo -> negativo: 6; positivo -> neutro: 1;
- negativo -> positivo: 3;
- neutro -> positivo: 4.

Artefatos locais:

- resumo: `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_121606_196690/reports/summary_metrics.md`;
- relatorio JSON: `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_121606_196690/reports/albertina_ptbr_100m/report.json`;
- predicoes e erros: `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_121606_196690/`;
- modelo: `../../data/models/etapa2_subsymbolic/transformers/20260612_121606_196690/albertina_ptbr_100m/`.

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

## analise de artefatos 20260612_131841_753357

Depois de consolidar as regexes de pistas superficiais em
`pln_core.eval.text_treatments`, a analise de artefatos foi reexecutada para
usar a mesma definicao dos diagnosticos de vazamento.

Principais achados no teste comum:

- split quase balanceado: 1667 positivos, 1666 negativos e 1666 neutros;
- duplicacao exata treino/teste muito baixa: 9 textos normalizados em comum,
  cobrindo 10 linhas do teste, sem conflito de rotulo;
- 99,28% dos tweets positivos contem emoticon positivo;
- 99,88% dos tweets negativos contem emoticon negativo;
- 99,70% dos tweets neutros contem URL.

Artefatos locais:

- resumo: `../../outputs/etapa2_subsymbolic/data_artifacts/20260612_131841_753357/reports/artifact_analysis.md`;
- tabelas: `../../outputs/etapa2_subsymbolic/data_artifacts/20260612_131841_753357/tables/`;
- figura: `../../outputs/etapa2_subsymbolic/data_artifacts/20260612_131841_753357/figures/cue_prevalence_test.png`.

## diagnostico de vazamento 20260612_131708_831350

Comando:

```bash
uv run python etapas/etapa2_subsimbolica/pipelines/run_leakage_diagnostics.py
```

Esse diagnostico confirma que a acuracia neural quase perfeita no split Kaggle
e majoritariamente explicavel por pistas superficiais do proprio protocolo de
supervisao distante. Uma Regressao Logistica treinada somente com
`has_positive_emoticon`, `has_negative_emoticon` e `has_url` obteve acuracia
`0.9970` e macro-F1 `0.9970` no teste bruto. Uma regra manual usando
emoticon negativo, depois emoticon positivo, depois URL obteve acuracia
`0.9958`.

Quando o teste tem emoticons e URLs removidos, o mesmo classificador cue-only
cai para acuracia `0.3333` e macro-F1 `0.1666`, como esperado em um split
balanceado sem essas pistas.

| Modelo | Treino | Teste | Acuracia | Macro-F1 |
| --- | --- | --- | ---: | ---: |
| `cue_only_logreg_raw_test` | raw | raw | 0.9970 | 0.9970 |
| `cue_rule_emoticon_url` | raw | raw | 0.9958 | 0.9958 |
| `tfidf_logreg` | raw | raw | 0.8172 | 0.8164 |
| `tfidf_logreg` | sem emoticons/URLs | sem emoticons/URLs | 0.8086 | 0.8094 |
| `tfidf_linear_svm` | raw | raw | 0.8080 | 0.8084 |
| `tfidf_linear_svm` | sem emoticons/URLs | sem emoticons/URLs | 0.8020 | 0.8030 |

Artefatos locais:

- resumo: `../../outputs/etapa2_subsymbolic/leakage_diagnostics/20260612_131708_831350/reports/summary_metrics.md`;
- tabela de pistas: `../../outputs/etapa2_subsymbolic/leakage_diagnostics/20260612_131708_831350/tables/cue_prevalence.csv`;
- figuras: `../../outputs/etapa2_subsymbolic/leakage_diagnostics/20260612_131708_831350/figures/`;
- predicoes e erros: `../../outputs/etapa2_subsymbolic/leakage_diagnostics/20260612_131708_831350/`.

Implicacao para o relatorio: resultados neurais brutos devem ser descritos como
validos para o split Kaggle original, mas fracos como evidencia de semantica de
sentimento. A tabela final da etapa 2 deve incluir resultados sem emoticons/URLs
e a baseline cue-only.

## transformer sem emoticons/URLs 20260612_132142_936441

Comando:

```bash
uv run --extra transformers python etapas/etapa2_subsimbolica/pipelines/run_transformer_benchmark.py model=distilbert_multilingual train_per_class=1000 model.training.epochs=1 trainer.use_cpu=true text_treatment=strip_emoticons_urls
```

Execucao em CPU com `distilbert_multilingual`, 1000 exemplos por classe no
treino, teste comum completo, 1 epoca, batch size 4, gradient accumulation 4 e
remocao de emoticons/URLs antes da tokenizacao.

| Modelo | Tratamento | Treino | Teste | Acuracia | Macro-F1 | F1 positivo | F1 negativo | F1 neutro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `distilbert_multilingual` | raw | 3000 | 4999 | 0.9950 | 0.9950 | 0.9925 | 0.9979 | 0.9946 |
| `distilbert_multilingual` | sem emoticons/URLs | 3000 | 4999 | 0.7465 | 0.7385 | 0.5785 | 0.6944 | 0.9425 |

Confusao principal sem emoticons/URLs:

- positivo -> negativo: 700; positivo -> neutro: 125;
- negativo -> positivo: 368; negativo -> neutro: 39;
- neutro -> positivo: 34; neutro -> negativo: 1.

Artefatos locais:

- resumo: `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_132142_936441/reports/summary_metrics.md`;
- relatorio JSON: `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_132142_936441/reports/distilbert_multilingual/report.json`;
- predicoes e erros: `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_132142_936441/`;
- modelo: `../../data/models/etapa2_subsymbolic/transformers/20260612_132142_936441/distilbert_multilingual/`.

Implicacao: remover os atalhos reduz a acuracia neural em aproximadamente
24,8 pontos percentuais e a macro-F1 em aproximadamente 25,7 pontos. Isso
confirma que a comparacao final deve separar "Kaggle bruto" de "robustez sem
pistas de rotulagem".
