# relatório da etapa 2

Espaço do relatório da segunda etapa. A versão LaTeX consolida introdução, arquitetura da solução estatística/neural, recursos e ferramentas, treino/teste, corpus, métricas, exemplos de execução, resultados, análise de erros, comparação com a etapa 1, limitações e instruções de execução.

A pasta `latex/` contém a versão principal em português e as figuras usadas na entrega.

Diagnósticos incorporados ao relatório:

- resultado bruto no split Kaggle, pois é a reprodução direta do corpus
  original;
- resultado sem emoticons/URLs para todos os sistemas em que a condicao foi
  executada, incluindo o `oplexicon_regex`;
- diagnóstico exploratório `strip_social_source_cues`, pois também remove
  menções, hashtags e marcadores de fontes neutras recorrentes.
