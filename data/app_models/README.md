# app_models

Artefatos pequenos e curados para a aplicação Streamlit implantada.

Esta pasta é versionada de propósito. Ela não substitui os artefatos
experimentais completos em `data/models/`; serve apenas para garantir que o
deploy do app tenha pelo menos os modelos leves necessários para demonstração.

Conteúdo atual:

- `etapa2_subsymbolic/20260614_113447_389024/tfidf_logreg.joblib`
- `etapa2_subsymbolic/20260614_113447_389024/tfidf_logreg.metadata.json`
- `etapa2_subsymbolic/20260614_113447_389024/tfidf_linear_svm.joblib`
- `etapa2_subsymbolic/20260614_113447_389024/tfidf_linear_svm.metadata.json`

Esses modelos foram treinados com `text_treatment=strip_emoticons_urls`, isto
é, sem emoticons e URLs, para evitar usar no app a condição bruta mais afetada
por vazamento de rótulo.
