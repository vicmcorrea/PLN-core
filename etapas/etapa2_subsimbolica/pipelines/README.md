# pipelines da etapa 2

Esta pasta guarda os scripts de treino e avaliação subsimbólica.

Os scripts daqui devem depender de código reutilizável em `src/pln_core`, mas não devem misturar resultados ou configurações com a etapa 1.

## suite clássica TF-IDF

O pipeline `run_classical_benchmark_suite.py` treina e avalia os baselines
classicos no corpus Kaggle comum:

```bash
uv run python etapas/etapa2_subsimbolica/pipelines/run_classical_benchmark_suite.py
```

Modelos executados por padrão:

- `tfidf_logreg`: TF-IDF + Regressao Logistica.
- `tfidf_linear_svm`: TF-IDF + Linear SVM.

Entradas:

- treino: `../../data/raw/portuguese-tweets-for-sentiment-analysis/TrainingDatasets/Train3Classes.csv`;
- teste comum: `../../data/raw/portuguese-tweets-for-sentiment-analysis/TestDatasets/Test3classes.csv`;
- config principal: `../configs/benchmark_suite.yaml`;
- configs de modelo: `../configs/model/`.

Saidas por execucao:

- relatorios e tabelas: `../../outputs/etapa2_subsymbolic/benchmark_suite/<run_id>/reports/`;
- predicoes: `../../outputs/etapa2_subsymbolic/benchmark_suite/<run_id>/predictions/`;
- casos de erro: `../../outputs/etapa2_subsymbolic/benchmark_suite/<run_id>/cases/`;
- figuras: `../../outputs/etapa2_subsymbolic/benchmark_suite/<run_id>/figures/`;
- modelos exportados: `../../data/models/etapa2_subsymbolic/<run_id>/`.

## ordem de evolução

1. preparação do corpus Kaggle;
2. treino TF-IDF + Regressão Logística;
3. treino TF-IDF + Linear SVM;
4. fine-tuning de transformer;
5. consolidação dos resultados para comparação com a etapa 1.
