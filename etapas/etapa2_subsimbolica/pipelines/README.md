# pipeline da etapa 2

Esta pasta será usada para os scripts de treino e avaliação subsimbólica.

Os scripts daqui devem depender de código reutilizável em `src/pln_core`, mas não devem misturar resultados ou configurações com a etapa 1.

Ordem prevista:

1. preparação do corpus Kaggle;
2. treino TF-IDF + Regressão Logística;
3. treino TF-IDF + Linear SVM;
4. fine-tuning de transformer;
5. consolidação dos resultados para comparação com a etapa 1.

Entradas planejadas:

- treino: `../../data/raw/portuguese-tweets-for-sentiment-analysis/TrainingDatasets/Train3Classes.csv`;
- teste comum: `../../data/raw/portuguese-tweets-for-sentiment-analysis/TestDatasets/Test3classes.csv`;
- configs: `../configs/`;
- saídas: `../../outputs/etapa2_subsymbolic/`.
