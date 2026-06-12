# resumo da suite classica da etapa 2

A primeira suite subsimbolica da etapa 2 foi executada no mesmo corpus usado
pela etapa 1: o dataset Kaggle `augustop/portuguese-tweets-for-sentiment-analysis`.
O treino usou `TrainingDatasets/Train3Classes.csv` com 100000 exemplos, e a
avaliacao usou `TestDatasets/Test3classes.csv` com 4999 exemplos. Essa divisao
mantem a comparacao direta com o baseline simbolico `oplexicon_regex`, cuja
execucao de referencia da etapa 1 e `20260612_152415_135433`.

A suite da etapa 2 treinou dois baselines classicos supervisionados. O primeiro
foi TF-IDF com Regressao Logistica balanceada, e o segundo foi TF-IDF com Linear
SVM balanceado. Ambos usaram n-gramas de palavras `(1, 2)`, `min_df=2`,
`max_df=0.95`, `sublinear_tf=true` e seed `42`.

| Modelo | Acuracia | Macro-F1 | F1 positivo | F1 negativo | F1 neutro |
| --- | ---: | ---: | ---: | ---: | ---: |
| `oplexicon_regex` | 0.5979 | 0.5960 | 0.6515 | 0.6407 | 0.4956 |
| `tfidf_logreg` | 0.8172 | 0.8164 | 0.7374 | 0.7421 | 0.9697 |
| `tfidf_linear_svm` | 0.8080 | 0.8084 | 0.7223 | 0.7351 | 0.9677 |

O melhor resultado classico inicial foi o TF-IDF + Regressao Logistica, com
macro-F1 `0.8164`. Esse valor fornece a primeira referencia subsimbolica para a
comparacao da etapa 2 e deve ser usado como baseline classico antes do
fine-tuning dos transformers. A alta pontuacao da classe neutra sugere que os
marcadores lexicais e padroes superficiais do corpus sao muito informativos para
essa classe, enquanto as classes positiva e negativa permanecem mais dificeis.

Depois dos diagnosticos de vazamento, a comparacao tratada passou a ser
obrigatoria. Na condicao `strip_emoticons_urls`, o `oplexicon_regex` cai para
acuracia `0.3697` e macro-F1 `0.3668`, enquanto o TF-IDF + Regressao Logistica
fica em macro-F1 `0.8094`. Portanto, a tabela bruta deve ser apresentada como
resultado no split Kaggle original, mas nao como unica evidencia de robustez.

Artefatos locais da execucao consolidada:

- `../../outputs/etapa2_subsymbolic/benchmark_suite/20260612_103137_624831/reports/summary_metrics.md`;
- `../../outputs/etapa2_subsymbolic/benchmark_suite/20260612_103137_624831/figures/`;
- `../../outputs/etapa2_subsymbolic/benchmark_suite/20260612_103137_624831/predictions/`;
- `../../outputs/etapa2_subsymbolic/benchmark_suite/20260612_103137_624831/cases/`;
- `../../data/models/etapa2_subsymbolic/20260612_103137_624831/`.
