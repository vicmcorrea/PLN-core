# etapa 2 subsimbólica

Esta pasta organiza a segunda etapa do projeto. A etapa 2 deve desenvolver uma solução subsimbólica, estatística e/ou neural, para o mesmo problema da etapa 1: análise de sentimentos em português brasileiro.

Pipelines, configs, experimentos e relatórios da etapa 2 ficam separados da etapa 1. Código reutilizável, como carregamento de dados e métricas, continua em `../../src/pln_core`.

## objetivo

Comparar a solução simbólica da etapa 1 com abordagens treinadas no mesmo corpus:

- TF-IDF + Regressão Logística como baseline simples.
- TF-IDF + Linear SVM como baseline clássico forte.
- Fine-tuning de transformer como versão neural principal.

## estrutura

- `configs/`: dataset comum, baselines TF-IDF e modelos neurais candidatos.
- `pipelines/`: futuros scripts de treino, avaliação e consolidação.
- `experiments/`: índice dos experimentos da etapa 2.
- `reports/`: espaço do relatório e LaTeX desta etapa.
- `especificacao_etapa2.md`: leitura da especificação oficial da segunda etapa.
- `plano_modelagem.md`: plano de execução técnico.

## corpus comum

O corpus recomendado é o Kaggle "Portuguese Tweets for Sentiment Analysis":

https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis

Arquivos planejados:

- `TrainingDatasets/Train3Classes.csv` para treino dos modelos estatísticos e neurais.
- `TestDatasets/Test3classes.csv` para avaliação comum contra a etapa 1.

## arquitetura planejada

1. Carregamento do corpus e padronização dos rótulos.
2. Pré-processamento mínimo e reprodutível.
3. Treino de baselines TF-IDF.
4. Fine-tuning de modelo neural pré-treinado.
5. Avaliação com as mesmas métricas da etapa 1.
6. Comparação quantitativa e qualitativa entre simbólico, estatístico e neural.

## modelos candidatos

- `sklearn.feature_extraction.text.TfidfVectorizer` + `LogisticRegression`.
- `sklearn.feature_extraction.text.TfidfVectorizer` + `LinearSVC`.
- `FacebookAI/xlm-roberta-base` como transformer principal.
- `PORTULAN/albertina-100m-portuguese-ptbr-encoder` como alternativa focada em português brasileiro.
- `distilbert/distilbert-base-multilingual-cased` como opção leve se houver tempo.

Referências de literatura/modelos a validar para o relatório: XLM-R/XLM-T para modelos multilingues em tweets, a família Albertina/PT-* para encoders de português, e artigos de corpora de sentimento em português brasileiro como contexto, sem transformar TweetSentBR no corpus operacional.

## critério de sucesso

A etapa 2 estará pronta quando tivermos:

- dataset carregado de forma reprodutível;
- baseline TF-IDF + Regressão Logística;
- baseline TF-IDF + Linear SVM;
- pelo menos um transformer fine-tuned;
- resultados salvos em formato comparável aos resultados simbólicos;
- análise de erros e acertos no relatório.
