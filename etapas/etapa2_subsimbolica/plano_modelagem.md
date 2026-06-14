# plano de modelagem da etapa 2

## sequência de implementação

1. Criar loader local do corpus Kaggle: concluido no nucleo compartilhado.
2. Rodar a solução simbólica da etapa 1 no teste comum: concluido com `oplexicon_regex`.
3. Separar os pipelines/configs/experimentos/relatórios da etapa 2 dos artefatos da etapa 1: concluido.
4. Implementar TF-IDF + Regressão Logística: implementado na suite classica.
5. Implementar TF-IDF + Linear SVM: implementado na suite classica.
6. Executar e registrar a suite classica da etapa 2.
7. Implementar treino/fine-tuning de `FacebookAI/xlm-roberta-base`: pipeline criado, execucao pendente.
8. Avaliar `PORTULAN/albertina-100m-portuguese-ptbr-encoder` se houver tempo de execução.
9. Consolidar tabela comparativa final.

## divisão treino/teste

Usar a divisão fornecida pelo próprio corpus:

- treino multiclasse: `TrainingDatasets/Train3Classes.csv`;
- teste multiclasse: `TestDatasets/Test3classes.csv`.

Caso o treino completo fique pesado para transformers, usar uma amostra estratificada para desenvolvimento e depois rodar o experimento final com o maior subconjunto viável.

## métricas

Todas as abordagens devem produzir:

- acurácia;
- macro-F1;
- F1 por classe;
- matriz de confusão;
- tempo de execução;
- exemplos representativos de erro.

## comparação no relatório

A comparação deve deixar explícito:

- a solução simbólica não aprende pesos a partir do treino;
- TF-IDF + Regressão Logística e TF-IDF + Linear SVM usam features esparsas de superfície;
- transformers usam representações contextuais pré-treinadas e são ajustados no corpus;
- o teste final é igual para todos.

## organização dos artefatos

- Configs da etapa 2: `configs/`.
- Pipelines: `pipelines/`.
- Logs, tabelas, predicoes e figuras da suite classica: `../../outputs/etapa2_subsymbolic/benchmark_suite/<run_id>/`.
- Logs, tabelas, predicoes e figuras transformer: `../../outputs/etapa2_subsymbolic/transformer_benchmark/<run_id>/`.
- Modelos treinados exportados: `../../data/models/etapa2_subsymbolic/<run_id>/`, quando forem pequenos o suficiente para armazenamento local.
- Checkpoints/modelos transformer: `../../data/models/etapa2_subsymbolic/transformers/<run_id>/<model>/`.
- Relatório e LaTeX: `reports/`.
- Dados brutos compartilhados: `../../data/raw/portuguese-tweets-for-sentiment-analysis/`.

## riscos

- O corpus Kaggle usa supervisão distante, então os rótulos podem conter ruído.
- Tweets têm gírias, ironia, abreviações e links, o que pode prejudicar métodos simbólicos.
- Transformers podem exigir GPU ou amostragem para caber no tempo disponível.
- A literatura com benchmark direto nesse dataset é limitada; por isso, os baselines internos precisam ser bem reprodutíveis.
