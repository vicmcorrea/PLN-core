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
| `distilbert_multilingual` sem emoticons/URLs | 3000 | 4999 | 0.7465 | 0.7385 | 0.5785 | 0.6944 | 0.9425 |
| `xlm_roberta_base` dev/full-test | 3000 | 4999 | 0.9968 | 0.9968 | 0.9952 | 0.9985 | 0.9967 |
| `xlm_roberta_base` sem emoticons/URLs | 3000 | 4999 | 0.7586 | 0.7494 | 0.5814 | 0.7117 | 0.9552 |
| `albertina_ptbr_100m` dev/full-test | 3000 | 4999 | 0.9972 | 0.9972 | 0.9958 | 0.9973 | 0.9985 |
| `albertina_ptbr_100m` sem emoticons/URLs | 3000 | 4999 | 0.7822 | 0.7808 | 0.6807 | 0.7005 | 0.9612 |

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

A primeira execucao transformer nessa condicao, `20260612_132142_936441`,
usou `distilbert_multilingual` com 1000 exemplos por classe no treino e removeu
emoticons/URLs tanto no treino quanto no teste. A acuracia caiu de `0,9950`
para `0,7465` e a macro-F1 caiu de `0,9950` para `0,7385`, confirmando que os
atalhos de rotulagem explicam grande parte do resultado bruto. A classe neutra
continua forte (`F1=0,9425`), enquanto positivo e negativo perdem muito mais
desempenho.

A execucao `20260612_132734_004665` repetiu a condicao sem emoticons/URLs com
`FacebookAI/xlm-roberta-base`, o modelo principal candidato. A acuracia caiu de
`0,9968` para `0,7586` e a macro-F1 caiu de `0,9968` para `0,7494`. O padrao de
erro permaneceu semelhante ao DistilBERT: neutro ficou alto (`F1=0,9552`), mas
positivo e negativo ficaram muito mais confundidos entre si.

A execucao `20260612_134601_518530` repetiu o mesmo diagnostico com
`PORTULAN/albertina-100m-portuguese-ptbr-encoder`. A acuracia caiu de `0,9972`
para `0,7822` e a macro-F1 caiu de `0,9972` para `0,7808`. Albertina ficou
melhor que DistilBERT e XLM-R nessa condicao controlada, mas a queda ainda e
grande o suficiente para confirmar que o split bruto mede principalmente as
pistas de supervisao distante.

Artefatos locais principais:

- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_113749_208484/reports/summary_metrics.md`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_115913_159120/reports/summary_metrics.md`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_121606_196690/reports/summary_metrics.md`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_134601_518530/reports/summary_metrics.md`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_113749_208484/predictions/distilbert_multilingual.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_115913_159120/predictions/xlm_roberta_base.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_121606_196690/predictions/albertina_ptbr_100m.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_134601_518530/predictions/albertina_ptbr_100m.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_113749_208484/cases/distilbert_multilingual_errors.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_115913_159120/cases/xlm_roberta_base_errors.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_121606_196690/cases/albertina_ptbr_100m_errors.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_132142_936441/reports/summary_metrics.md`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_132734_004665/reports/summary_metrics.md`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_132142_936441/predictions/distilbert_multilingual.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_132734_004665/predictions/xlm_roberta_base.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_132142_936441/cases/distilbert_multilingual_errors.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_132734_004665/cases/xlm_roberta_base_errors.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_134601_518530/cases/albertina_ptbr_100m_errors.csv`;
- `../../outputs/etapa2_subsymbolic/transformer_benchmark/20260612_113749_208484/figures/confusion_distilbert_multilingual.png`;
- `../../data/models/etapa2_subsymbolic/transformers/20260612_113749_208484/distilbert_multilingual/`.
- `../../data/models/etapa2_subsymbolic/transformers/20260612_132142_936441/distilbert_multilingual/`.
- `../../data/models/etapa2_subsymbolic/transformers/20260612_115913_159120/xlm_roberta_base/`.
- `../../data/models/etapa2_subsymbolic/transformers/20260612_132734_004665/xlm_roberta_base/`.
- `../../data/models/etapa2_subsymbolic/transformers/20260612_121606_196690/albertina_ptbr_100m/`.
- `../../data/models/etapa2_subsymbolic/transformers/20260612_134601_518530/albertina_ptbr_100m/`.
- `../../outputs/etapa2_subsymbolic/data_artifacts/20260612_131841_753357/reports/artifact_analysis.md`.
- `../../outputs/etapa2_subsymbolic/leakage_diagnostics/20260612_131708_831350/reports/summary_metrics.md`.
