# pipelines da etapa 1

`run_symbolic_evaluation.py` executa uma combinacao dataset/analisador.
`run_symbolic_benchmark_suite.py` executa o baseline oficial `oplexicon_regex`
no corpus Kaggle compartilhado e salva perfil do dataset, predicoes, metricas,
casos de erro e figuras em uma pasta com `run_id`.

Comando principal da bateria:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_benchmark_suite.py
```

A bateria baixa o dataset Kaggle configurado se os arquivos esperados ainda nao
existirem em `data/raw/portuguese-tweets-for-sentiment-analysis/`. Para isso,
as credenciais do Kaggle precisam estar configuradas em `~/.kaggle/kaggle.json`
ou variaveis de ambiente equivalentes.

Comando individual:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py
```

Comando rapido:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py dataset=sample
```
