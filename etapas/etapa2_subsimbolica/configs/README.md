# configs da etapa 2

Configuracoes planejadas para os experimentos estatisticos e neurais.

- `default.yaml`: raiz Hydra futura da etapa 2.
- `dataset/kaggle_portuguese_tweets.yaml`: corpus comum das duas etapas.
- `model/tfidf_logreg.yaml`: baseline simples.
- `model/tfidf_linear_svm.yaml`: baseline classico forte.
- `model/xlm_roberta_base.yaml`: transformer principal.
- `model/albertina_ptbr_100m.yaml`: encoder portugues candidato.
- `model/distilbert_multilingual.yaml`: transformer leve opcional.

Os scripts ainda serao implementados em `../pipelines/`.
