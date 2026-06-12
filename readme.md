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

Coloque os arquivos extraídos em:

```text
data/raw/portuguese-tweets-for-sentiment-analysis/
```

Arquivos esperados:

- `TrainingDatasets/Train3Classes.csv`
- `TestDatasets/Test3classes.csv`

Se usar o cliente Kaggle, baixe o dataset `augustop/portuguese-tweets-for-sentiment-analysis` e extraia os arquivos nessa pasta. Se preferir, baixe pelo navegador e mantenha exatamente a mesma estrutura de subpastas.

## comandos principais

Instalar dependências:

```bash
uv sync
```

Rodar Streamlit:

```bash
uv run streamlit run streamlit_app.py
```

Rodar avaliação simbólica no corpus comum:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py
```

Rodar avaliação rápida no dataset didático:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py dataset=sample
```

Os resultados experimentais são salvos em `outputs/`, que fica ignorado pelo Git. A etapa 2 ainda está em estruturação: seus modelos e pipelines ficam em `etapas/etapa2_subsimbolica/`.
