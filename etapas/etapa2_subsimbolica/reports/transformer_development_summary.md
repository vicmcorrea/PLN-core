# resumo de desenvolvimento transformer

O pipeline transformer da etapa 2 foi validado com
`distilbert/distilbert-base-multilingual-cased`, usando o mesmo carregador do
corpus Kaggle empregado na etapa 1 e na suite classica TF-IDF. As execucoes
abaixo foram feitas em CPU para evitar erros de memoria compartilhada no backend
MPS local. Elas servem como validacao de pipeline e como resultados neurais de
desenvolvimento; o resultado final ainda deve ser obtido com uma execucao
consolidada, idealmente com mais dados de treino e registro de hardware/tempo.

| Sistema | Treino | Teste | Acuracia | Macro-F1 | F1 positivo | F1 negativo | F1 neutro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `oplexicon_regex` | 0 | 4999 | 0.5979 | 0.5960 | 0.6515 | 0.6407 | 0.4956 |
| `tfidf_logreg` | 100000 | 4999 | 0.8172 | 0.8164 | 0.7374 | 0.7421 | 0.9697 |
| `cue_only_logreg` | 100000 | 4999 | 0.9970 | 0.9970 | 0.9961 | 0.9991 | 0.9958 |
| `distilbert_multilingual` dev | 3000 | 999 | 0.9970 | 0.9970 | 0.9955 | 0.9985 | 0.9970 |
| `distilbert_multilingual` dev/full-test | 3000 | 4999 | 0.9950 | 0.9950 | 0.9925 | 0.9979 | 0.9946 |
| `xlm_roberta_base` dev/full-test | 3000 | 4999 | 0.9968 | 0.9968 | 0.9952 | 0.9985 | 0.9967 |
| `albertina_ptbr_100m` dev/full-test | 3000 | 4999 | 0.9972 | 0.9972 | 0.9958 | 0.9973 | 0.9985 |

A execucao `20260612_113749_208484` e a primeira execucao neural de
desenvolvimento avaliada no teste comum completo. Ela treinou com 1000 exemplos
por classe da particao de treino e avaliou nos 4999 tweets da particao de teste.
As execucoes `20260612_115913_159120` e `20260612_121606_196690` repetem o
mesmo desenho experimental com XLM-R base e Albertina-100M pt-BR. Como todos os
resultados neurais sao muito altos, o relatorio final deve apresentar uma
checagem de possiveis duplicatas entre treino e teste e de artefatos lexicais
associados aos rotulos antes de interpretar esses valores como evidencia
definitiva de generalizacao.

Uma checagem inicial de duplicatas exatas normalizadas encontrou 99043 textos
unicos no treino, 4996 textos unicos no teste e apenas 9 textos unicos em comum
entre treino e teste, cobrindo 10 linhas do teste. Todos esses casos tinham
rotulos consistentes entre as particoes. Portanto, duplicacao exata entre treino
e teste nao parece explicar, sozinha, o desempenho alto.

A analise de artefatos `20260612_131841_753357` confirmou que o corpus possui
sinais superficiais muito fortes: no teste comum, 99,28% dos tweets positivos
contem emoticon positivo, 99,88% dos tweets negativos contem emoticon negativo
e 99,70% dos tweets neutros contem URL. O diagnostico de vazamento
`20260612_131708_831350` mostra que uma Regressao Logistica com apenas
`has_positive_emoticon`, `has_negative_emoticon` e `has_url` chega a acuracia
`0,9970` e macro-F1 `0,9970`. Portanto, os resultados transformer brutos sao
validos para o split Kaggle original, mas nao devem ser apresentados como prova
forte de aprendizagem semantica sem uma condicao sem emoticons/URLs.

Artefatos locais principais:

- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_113749_208484/reports/summary_metrics.md`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_115913_159120/reports/summary_metrics.md`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_121606_196690/reports/summary_metrics.md`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_113749_208484/predictions/distilbert_multilingual.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_115913_159120/predictions/xlm_roberta_base.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_121606_196690/predictions/albertina_ptbr_100m.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_113749_208484/cases/distilbert_multilingual_errors.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_115913_159120/cases/xlm_roberta_base_errors.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_121606_196690/cases/albertina_ptbr_100m_errors.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_113749_208484/figures/confusion_distilbert_multilingual.png`;
- `../../data/models/etapa2_subsymbolic/transformers/20260612_113749_208484/distilbert_multilingual/`.
- `../../data/models/etapa2_subsymbolic/transformers/20260612_115913_159120/xlm_roberta_base/`.
- `../../data/models/etapa2_subsymbolic/transformers/20260612_121606_196690/albertina_ptbr_100m/`.
- `../../outputs/etapa2_subsymbolic/data_artifacts/20260612_131841_753357/reports/artifact_analysis.md`.
- `../../outputs/etapa2_subsymbolic/leakage_diagnostics/20260612_131708_831350/reports/summary_metrics.md`.
