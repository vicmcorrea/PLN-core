# pipelines da etapa 1

`run_symbolic_evaluation.py` executa uma combinacao dataset/analisador.
`run_symbolic_benchmark_suite.py` executa o baseline oficial `oplexicon_regex`
no corpus Kaggle compartilhado e salva perfil do dataset, predicoes, metricas,
casos de erro e figuras em uma pasta com `run_id`.

Comando principal da bateria:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_benchmark_suite.py
```

Para repetir o baseline simbolico nas mesmas versoes de texto usadas na Etapa
2, execute a bateria com a lista de tratamentos:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_benchmark_suite.py \
  'text_treatments=[raw,strip_emoticons_urls,strip_social_source_cues]'
```

Cada tratamento gera predicoes, casos de erro, relatorios e matrizes de
confusao com chave propria, por exemplo
`oplexicon_regex__strip_emoticons_urls`, evitando sobrescrita entre condicoes.

A bateria baixa o dataset Kaggle configurado se os arquivos esperados ainda nao
existirem em `data/raw/portuguese-tweets-for-sentiment-analysis/`. Para isso,
as credenciais do Kaggle precisam estar configuradas em `~/.kaggle/kaggle.json`
ou variaveis de ambiente equivalentes.

Comando individual:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py
```

Para uma avaliacao individual com texto tratado:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py \
  dataset.kwargs.text_treatment=strip_emoticons_urls
```

Comando rapido:

```bash
uv run python etapas/etapa1_simbolica/pipelines/run_symbolic_evaluation.py dataset=sample
```
