# pln-core

Projeto de análise de sentimentos em português brasileiro para a disciplina SCC5908.

## organização do repositório

O diretório raiz fica reservado para código compartilhado, dados compartilhados, testes e a aplicação Streamlit:

- `src/pln_core/`: núcleo reutilizável, solução simbólica, carregadores de dados e harness de avaliação.
- `data/`: recursos compartilhados e local esperado para o corpus comum. Dados grandes ficam fora do Git.
- `streamlit_app.py`: interface principal do projeto.
- `tests/`: testes automatizados do núcleo compartilhado.
- `etapas/etapa1_simbolica/`: primeira etapa, solução simbólica revisada e corrigida.
- `etapas/etapa2_subsimbolica/`: segunda etapa, solução estatística/neural.

Cada etapa possui seus próprios `configs/`, `pipelines/`, `experiments/` e `reports/`. A CLI foi removida do fluxo ativo; os pontos de entrada suportados são o Streamlit e os pipelines dentro de `etapas/`.

## corpus comum

As duas etapas devem usar o mesmo corpus principal:

https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis

Os CSVs do Kaggle nao ficam versionados no Git e tambem nao sao empacotados no
arquivo de entrega. Depois de instalar as dependencias, baixe e organize o
corpus com o script abaixo. Ele usa o cliente oficial do Kaggle, entao as
credenciais precisam estar configuradas em `~/.kaggle/kaggle.json` ou em
variaveis de ambiente equivalentes.

```bash
uv run python scripts/download_kaggle_dataset.py
```

O local padrao criado pelo script e:

```text
data/raw/portuguese-tweets-for-sentiment-analysis/
```

Arquivos esperados:

- `TrainingDatasets/Train3Classes.csv`
- `TestDatasets/Test3classes.csv`

Se preferir baixar pelo navegador, use o dataset
`augustop/portuguese-tweets-for-sentiment-analysis` e mantenha exatamente a
mesma estrutura de subpastas.

## comandos principais

Instalar dependências:

```bash
uv sync
```

Baixar e organizar o corpus comum:

```bash
uv run python scripts/download_kaggle_dataset.py
```

Rodar Streamlit:

```bash
uv run streamlit run streamlit_app.py
```

O Streamlit descobre primeiro os modelos leves versionados em
`data/app_models/etapa2_subsymbolic/` e, em seguida, artefatos locais ignorados
em `data/models/etapa2_subsymbolic/`. Quando existir um artefato
`tfidf_logreg` treinado com `strip_emoticons_urls`, ele vira o modelo padrão do
app. O `oplexicon_regex` continua disponível como opção simbólica, e os
artefatos TF-IDF brutos ficam selecionáveis apenas para diagnóstico local do
split Kaggle original.

Rodar o baseline simbolico oficial (`oplexicon_regex`) no corpus comum, com
perfil do dataset, metricas, predicoes, casos de erro e figuras para o texto
bruto e para as condicoes tratadas usadas na etapa 2:

```bash
uv run python \
  etapas/etapa1_simbolica/pipelines/run_symbolic_benchmark_suite.py \
  'text_treatments=[raw,strip_emoticons_urls,strip_social_source_cues]'
```

Rodar uma avaliacao simbolica individual no corpus comum:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py
```

Rodar os baselines classicos da etapa 2 no mesmo split Kaggle usado pela etapa 1:

```bash
uv run python etapas/etapa2_subsimbolica/pipelines/run_classical_benchmark_suite.py
```

Exportar os modelos TF-IDF tratados que o Streamlit deve usar por padrão:

```bash
uv run python etapas/etapa2_subsimbolica/pipelines/run_classical_benchmark_suite.py \
  text_treatment=strip_emoticons_urls \
  symbolic_baseline.text_treatment=strip_emoticons_urls \
  symbolic_baseline.accuracy=0.36967393478695737 \
  symbolic_baseline.macro_f1=0.3667637673993909 \
  symbolic_baseline.positive_f1=0.3958817668548655 \
  symbolic_baseline.negative_f1=0.3174382178907066 \
  symbolic_baseline.neutral_f1=0.3869713174526009
```

Rodar um smoke test transformer da etapa 2:

```bash
uv sync --extra transformers
uv run --extra transformers python etapas/etapa2_subsimbolica/pipelines/run_transformer_benchmark.py model=distilbert_multilingual train_max_examples=120 test_max_examples=60 model.training.epochs=1 trainer.use_cpu=true
```

Rodar avaliação rápida no dataset didático:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py dataset=sample
```

Os resultados experimentais são salvos em `outputs/`, que fica ignorado pelo Git. Cada execução usa um `run_id` com timestamp para evitar sobrescrever resultados anteriores:

```text
outputs/etapa1_symbolic/runs/<run_id>/<dataset>/<analyzer>/
outputs/etapa1_symbolic/benchmark_suite/<run_id>/
outputs/etapa2_subsymbolic/benchmark_suite/<run_id>/
outputs/etapa2_subsymbolic/transformer_benchmark/<run_id>/
```

A etapa 2 ja possui uma suite classica inicial com TF-IDF + Regressao
Logistica e TF-IDF + Linear SVM, diagnosticos de vazamento por pistas
superficiais e um pipeline transformer opcional para fine-tuning.
