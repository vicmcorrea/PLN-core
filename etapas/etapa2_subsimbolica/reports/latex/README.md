# LaTeX da etapa 2

Esta pasta contem o relatorio em LaTeX da segunda etapa. O texto segue o
formato do relatorio revisado da Etapa 1 e incorpora a comparacao entre a
solucao simbolica, baselines TF-IDF, diagnosticos de vazamento e execucoes
transformer de desenvolvimento.

As figuras e tabelas atuais ja incorporam a execucao simbolica
`20260612_152415_135433`, incluindo os resultados `raw`,
`strip_emoticons_urls` e `strip_social_source_cues`.

O texto evita referencias a organizacao interna do repositorio fora da secao de
execucao. A parte metodologica agora explica, em linguagem de relatorio, o papel
da solucao simbolica, do TF-IDF, da Regressao Logistica, do SVM linear, dos
transformers e do diagnostico de vazamento.

A versao atual tambem explicita a origem das escolhas experimentais. O relatorio
separa decisoes vindas do corpus, como o split fixo de treino e teste, decisoes
baseadas na literatura, como o uso de TF-IDF com modelos lineares, e limitacoes
praticas, como as rodadas neurais em amostra estratificada de desenvolvimento.
As novas citacoes foram verificadas antes de entrar no `references.bib`.

A revisao mais recente reforca os itens pedidos na especificacao da Etapa 2. O
texto agora traz uma tabela de recursos e ferramentas com tamanho e papel de
cada recurso, alem de uma ilustracao de execucao ponta a ponta para exemplos
curtos antes da secao de resultados.

A tabela de exemplos qualitativos foi preenchida com casos reais extraidos das
predicoes salvas. Ela cobre falhas do simbolico corrigidas pelo TF-IDF, falhas
do TF-IDF em textos ambiguos ou ironicos, um exemplo com Albertina 100M pt-BR,
e um exemplo em que o modelo bruto parece melhor apenas porque a URL atua como
pista de rotulo. A aplicacao Streamlit usa TF-IDF + Regressao Logistica tratado
como modo padrao; os transformers permanecem como modelos de benchmark e
discussao no relatorio.

Arquivos principais:

- `main.tex`: estrutura geral do relatorio em portugues.
- `references.bib`: referencias iniciais para corpus, baselines, transformers e
  diagnosticos de artefatos.
- `scripts/build_figures.py`: gera as figuras locais do relatorio.
- `figures/`: figuras PDF/PNG usadas pelo LaTeX.

Para regenerar as figuras:

```bash
uv run python etapas/etapa2_subsimbolica/reports/latex/scripts/build_figures.py
```

Para compilar o relatorio:

```bash
cd etapas/etapa2_subsimbolica/reports/latex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Arquivos auxiliares de compilacao e PDFs gerados continuam ignorados pelo Git.
