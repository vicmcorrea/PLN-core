# plano de modelagem da etapa 2

## sequência de implementação

1. Criar loader local do corpus Kaggle.
2. Rodar a solução simbólica da etapa 1 no teste comum.
3. Separar os pipelines/configs/experimentos/relatórios da etapa 2 dos artefatos da etapa 1.
4. Implementar TF-IDF + Regressão Logística.
5. Implementar TF-IDF + Linear SVM.
6. Implementar treino/fine-tuning de `FacebookAI/xlm-roberta-base`.
7. Avaliar `PORTULAN/albertina-100m-portuguese-ptbr-encoder` se houver tempo de execução.
8. Consolidar tabela comparativa final.

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
- Pipelines futuros: `pipelines/`.
- Logs, checkpoints e tabelas locais: `../../outputs/etapa2_subsymbolic/`.
- Relatório e LaTeX: `reports/`.
- Dados brutos compartilhados: `../../data/raw/portuguese-tweets-for-sentiment-analysis/`.

## riscos

- O corpus Kaggle usa supervisão distante, então os rótulos podem conter ruído.
- Tweets têm gírias, ironia, abreviações e links, o que pode prejudicar métodos simbólicos.
- Transformers podem exigir GPU ou amostragem para caber no tempo disponível.
- A literatura com benchmark direto nesse dataset é limitada; por isso, os baselines internos precisam ser bem reprodutíveis.
