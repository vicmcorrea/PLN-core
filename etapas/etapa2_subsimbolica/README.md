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
- `pipelines/`: scripts de treino, avaliação, diagnostico de vazamento e consolidação.
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

O carregamento e o download opcional usam a configuração Hydra
`configs/dataset/kaggle_portuguese_tweets.yaml`.

## suite clássica atual

A suite inicial da etapa 2 treina e avalia os dois baselines TF-IDF no mesmo
split usado pela etapa 1:

```bash
uv run python etapas/etapa2_subsimbolica/pipelines/run_classical_benchmark_suite.py
```

Cada execução cria uma pasta com `run_id` próprio em
`../../outputs/etapa2_subsymbolic/benchmark_suite/<run_id>/`, contendo
configuração resolvida, manifesto do dataset, métricas por modelo, tabela
comparativa, predições, casos de erro e figuras. Os modelos `.joblib` ficam em
`../../data/models/etapa2_subsymbolic/<run_id>/`, que também é ignorado pelo Git.

## pipeline transformer

O pipeline `pipelines/run_transformer_benchmark.py` prepara o mesmo corpus para
fine-tuning de encoders transformer com `AutoModelForSequenceClassification`.
As dependencias pesadas ficam no extra opcional `transformers`:

```bash
uv sync --extra transformers
uv run --extra transformers python etapas/etapa2_subsimbolica/pipelines/run_transformer_benchmark.py model=distilbert_multilingual train_max_examples=120 test_max_examples=60 model.training.epochs=1 trainer.use_cpu=true
```

A execucao final deve remover os limites `train_max_examples` e
`test_max_examples`, usando o mesmo treino e teste da suite classica. Os
resultados ficam em `../../outputs/etapa2_subsymbolic/transformer_benchmark/<run_id>/`
e os checkpoints/modelos em `../../data/models/etapa2_subsymbolic/transformers/<run_id>/`.
Em Mac com MPS, use lotes pequenos ou `trainer.use_cpu=true` para testes
rapidos se houver erro de memoria compartilhada.

## diagnostico de vazamento

Como o corpus Kaggle foi criado por supervisao distante, emoticons e URLs podem
revelar o rotulo. O pipeline `pipelines/run_leakage_diagnostics.py` mede esse
efeito com baselines cue-only e com comparacoes raw versus texto sem
emoticons/URLs:

```bash
uv run python etapas/etapa2_subsimbolica/pipelines/run_leakage_diagnostics.py
```

A execucao `20260612_131708_831350` mostrou que uma Regressao Logistica com
apenas `has_positive_emoticon`, `has_negative_emoticon` e `has_url` atinge
acuracia `0.9970` no teste bruto. Portanto, resultados transformer brutos
devem ser reportados junto com essa limitacao e com uma condicao sem
emoticons/URLs.

## arquitetura planejada

1. Carregamento do corpus e padronização dos rótulos.
2. Pré-processamento mínimo e reprodutível.
3. Treino de baselines TF-IDF.
4. Fine-tuning de modelo neural pré-treinado.
5. Avaliação com as mesmas métricas da etapa 1.
6. Comparação quantitativa e qualitativa entre simbólico, estatístico e neural.
7. Diagnostico de pistas superficiais e condicao de robustez sem emoticons/URLs.

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
- pelo menos um transformer fine-tuned em execucao consolidada;
- resultados salvos em formato comparável aos resultados simbólicos;
- análise de erros e acertos no relatório.
