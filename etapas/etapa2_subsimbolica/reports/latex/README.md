# LaTeX da etapa 2

Esta pasta contem o rascunho estrutural do relatorio da segunda etapa. O texto
segue o formato do relatorio revisado da Etapa 1, mas ainda nao deve ser tratado
como versao final: faltam selecionar exemplos reais de sucesso/insucesso,
decidir o modelo que entrara na interface Streamlit e revisar as referencias
bibliograficas finais.

As figuras e tabelas atuais ja incorporam a execucao simbolica
`20260612_152415_135433`, incluindo os resultados `raw`,
`strip_emoticons_urls` e `strip_social_source_cues`.

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
