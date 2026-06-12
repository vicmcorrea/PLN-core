# configs da etapa 2

Configuracoes para os experimentos estatisticos e neurais.

- `default.yaml`: raiz Hydra futura da etapa 2.
- `benchmark_suite.yaml`: suite classica TF-IDF.
- `transformer_benchmark.yaml`: fine-tuning transformer com o mesmo split Kaggle.
- `leakage_diagnostics.yaml`: baselines cue-only e comparacoes raw vs sem emoticons/URLs.
- `dataset/kaggle_portuguese_tweets.yaml`: corpus comum das duas etapas.
- `model/tfidf_logreg.yaml`: baseline simples.
- `model/tfidf_linear_svm.yaml`: baseline classico forte.
- `model/xlm_roberta_base.yaml`: transformer principal.
- `model/albertina_ptbr_100m.yaml`: encoder portugues candidato.
- `model/distilbert_multilingual.yaml`: transformer leve opcional.

Os scripts correspondentes ficam em `../pipelines/`.

Os configs de benchmark aceitam `text_treatment=raw` por padrao. Para a
condicao de robustez principal, use `text_treatment=strip_emoticons_urls`.
