# referencias candidatas para a etapa 2

Estas referencias sao candidatas iniciais para orientar o relatorio. Elas ainda precisam ser validadas e convertidas para BibTeX antes da versao final.

- XLM-T: modelo multilingue em dominio Twitter para analise de sentimentos e tarefas relacionadas. Link inicial: https://hf.co/papers/2104.12250
- XLM-R e variantes: familia de modelos multilingues relevantes para fine-tuning. Link inicial: https://hf.co/papers/2105.00572
- Albertina PT-*: familia de encoders neurais para portugues. Link inicial: https://hf.co/papers/2305.06721
- PORTULAN ExtraGLUE: benchmark e modelos neurais para portugues. Link inicial: https://hf.co/papers/2404.05333
- TweetSentBR: corpus de tweets em portugues brasileiro util para contextualizacao bibliografica, sem ser usado como corpus operacional deste projeto. Link inicial: https://hf.co/papers/1712.08917

## supervisao distante, emoticons e artefatos de dataset

Buscas academicas via Valyu, Exa e Tavily em 12 de junho de 2026 encontraram
referencias diretamente relevantes para explicar o resultado muito alto dos
transformers no split Kaggle:

- Go, Bhayani e Huang (2009), "Twitter Sentiment Classification using Distant
  Supervision". Referencia classica para rotulagem distante de tweets por
  emoticons. O relatorio deve usar essa referencia para explicar que emoticons
  funcionam como sinais de rotulo, nao como anotacao humana independente.
- Wang e Castanon (2015), "Sentiment expression via emoticons on social media".
  DOI: https://doi.org/10.1109/bigdata.2015.7364034. O artigo mostra que
  emoticons sao sinais fortes de polaridade e que remover emoticons muda
  bastante o desempenho de classificadores.
- Yin, Alkhalifa e Zubiaga (2021), "The emojification of sentiment on social
  media". DOI: https://doi.org/10.48550/arxiv.2108.13898. O artigo descreve a
  coleta de datasets por supervisao distante com emojis/emoticons e explicita
  o procedimento de remover o simbolo usado como evidencia antes de tratar o
  texto restante como exemplo rotulado.
- Gururangan et al. (2018), "Annotation Artifacts in Natural Language Inference
  Data". ACL Anthology: https://aclanthology.org/N18-2017/. Embora seja NLI, e
  uma referencia metodologica importante para discutir artefatos de anotacao e
  baselines que exploram apenas pistas superficiais.
- Poliak et al. (2018), "Hypothesis Only Baselines in Natural Language
  Inference". ACL Anthology: https://aclanthology.org/S18-2023/. Referencia
  metodologica para justificar baselines degeneradas como diagnostico de vieses
  do dataset.

Antes da versao final, converter essas referencias para BibTeX e citar junto
com os resultados de `leakage_diagnostics`.
