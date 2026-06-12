# relatorio da etapa 2

Espaco do relatorio da segunda etapa. A especificacao pede aproximadamente 10 paginas com introducao, arquitetura da solucao estatistica/neural, recursos e ferramentas, treino/teste, corpus, metricas, exemplos de execucao, resultados, analise de erros, comparacao com a etapa 1, limitacoes e instrucoes de execucao.

A pasta `latex/` fica reservada para a versao LaTeX. As referencias bibliograficas devem ser validadas antes da escrita final.

Resumo ja consolidado para a escrita futura:

- `classical_benchmark_summary.md`: primeira comparacao entre o baseline simbolico da etapa 1 e os baselines TF-IDF da etapa 2 no split Kaggle comum.
- `transformer_development_summary.md`: validacao do pipeline transformer e primeiro resultado neural de desenvolvimento no teste comum.
- `dataset_artifact_analysis_summary.md`: analise de duplicatas, emoticons,
  URLs, mencoes, hashtags e termos associados aos rotulos no corpus comum.
- `latex/`: estrutura LaTeX da entrega da etapa 2, em portugues, seguindo o
  mesmo formato geral do relatorio revisado da etapa 1.

Diagnosticos que devem aparecer no relatorio:

- resultado bruto no split Kaggle, pois e a comparacao direta entre etapa 1,
  TF-IDF e modelos neurais;
- resultado sem emoticons/URLs, pois remove as pistas de supervisao distante
  mais obvias;
- diagnostico exploratorio `strip_social_source_cues`, pois tambem remove
  mencoes, hashtags e marcadores de fontes neutras recorrentes.
