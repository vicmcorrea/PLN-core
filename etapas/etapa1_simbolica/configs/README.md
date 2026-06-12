# configs da etapa 1

Configuracoes Hydra da avaliacao simbolica.

- `default.yaml`: avaliacao principal da etapa 1 no corpus Kaggle.
- `dataset/`: datasets disponiveis para avaliacao simbolica.
- `analyzer/`: variacoes simbolicas e baselines de referencia.

O dataset principal e `kaggle_tweets`. O dataset `sample` fica apenas como smoke test.

O loader `kaggle_tweets` aceita `dataset.kwargs.text_treatment`, com os mesmos
valores usados nos diagnosticos da Etapa 2: `raw`, `strip_emoticons_urls` e
`strip_social_source_cues`. A suite principal tambem aceita
`text_treatments=[...]` para avaliar o `oplexicon_regex` em varias condicoes sem
sobrescrever artefatos.
