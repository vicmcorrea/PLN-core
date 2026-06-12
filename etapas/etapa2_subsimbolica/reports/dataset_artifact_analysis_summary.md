# resumo da analise de artefatos do corpus

Execucao: `20260612_115649_211081`.

Comando:

```bash
uv run python etapas/etapa2_subsimbolica/pipelines/run_data_artifact_analysis.py
```

O pipeline analisa o mesmo corpus Kaggle usado pela etapa 1 simbolica, pela
suite TF-IDF e pelos experimentos transformer. Os artefatos gerados ficam em
`../../outputs/etapa2_subsymbolic/data_artifacts/20260612_115649_211081/`,
separados dos benchmarks de modelos.

## perfil do split

| Split | Linhas | Textos unicos | Duplicatas | Positivo | Negativo | Neutro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| treino | 100000 | 99043 | 957 | 33334 | 33333 | 33333 |
| teste | 4999 | 4996 | 3 | 1667 | 1666 | 1666 |

Duplicatas exatas normalizadas entre treino e teste: 9 textos unicos, cobrindo
10 linhas do teste. Nao houve conflito de rotulo nesses casos.

## pistas lexicais no teste

| Pista | Positivo | Negativo | Neutro |
| --- | ---: | ---: | ---: |
| emoticon positivo | 0,9142 | 0,0000 | 0,0006 |
| emoticon negativo | 0,0054 | 0,9988 | 0,0000 |
| URL | 0,2346 | 0,1849 | 0,9970 |
| mencao | 0,5135 | 0,4724 | 0,0222 |
| hashtag | 0,0312 | 0,0132 | 0,0900 |

Os termos mais associados a cada classe tambem refletem essa supervisao
distante. Positivos incluem `:)`, `:d`, `:p` e `:-)`. Negativos incluem `:(` e
`:-(`. Neutros incluem fontes e agregadores como `feedly`, `esportefera`,
`estadaoeconomia`, `g1sp` e `cbn`.

## uso no relatorio

Essa analise deve aparecer como uma limitacao experimental da etapa 2. A baixa
duplicacao exata sugere que o resultado neural alto nao vem apenas de vazamento
literal treino/teste. Ao mesmo tempo, emoticons e URLs sao sinais altamente
preditores dos rotulos, entao a comparacao entre simbolico, TF-IDF e
transformers deve discutir a dependencia desses marcadores superficiais.
