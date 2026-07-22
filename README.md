# Deteccao de Fraude em Transacoes de Cartao de Credito

## Descricao

Este projeto implementa um sistema de deteccao de fraude em transacoes de
cartao de credito usando machine learning, com foco em metricas honestas
(sem vazamento de dados) para um problema fortemente desbalanceado
(~0.5% de fraude).

## Estrutura do Projeto

```
Fraud_case/
├── config/
│   └── config.yaml            # fonte unica: colunas, paths, hiperparametros, seed
├── data/                       # fora do git (ver .gitignore)
│   ├── raw/                    # fraudTrain.csv / fraudTest.csv
│   └── processed/              # Parquet gerado por scripts/prepare_data.py
├── models/                     # pipeline de features + modelos treinados (.joblib)
├── notebooks/                  # analise exploratoria (nao usado em producao)
├── reports/                    # metricas, graficos, resumo do melhor modelo
├── src/fraud_case/
│   ├── config.py               # carregamento tipado de config.yaml
│   ├── data/                   # loader.py, balance.py
│   ├── features/               # engineering.py (sem vazamento), pipeline.py (sklearn)
│   ├── models/                 # train.py, evaluate.py, explain.py (SHAP)
│   └── utils/                  # geo.py, datetime_features.py, io.py, plotting.py
├── scripts/                    # CLIs finos: prepare_data.py, train_models.py, evaluate_models.py
├── tests/                      # pytest (unitarios + smoke test com dados reais)
├── main.py                     # orquestra os 3 scripts em sequencia
├── pyproject.toml              # ruff + pytest
└── requirements.txt
```

## Requisitos

- Python 3.10+

```bash
pip install -r requirements.txt
```

## Como Usar

### 1. Preparacao

Coloque `fraudTrain.csv` e `fraudTest.csv` em `data/raw/` (nao versionados no git).

### 2. Execucao

```bash
python main.py
```

Ou cada etapa separadamente:

```bash
python scripts/prepare_data.py      # feature engineering, sem vazamento, salva Parquet
python scripts/train_models.py      # ajusta o pipeline de features + treina os modelos
python scripts/evaluate_models.py   # avalia no teste, escolhe o melhor modelo por PR-AUC
```

Rodar os testes:

```bash
pytest
```

### 3. Configuracao

Todas as listas de colunas, hiperparametros de modelo, estrategia de
desbalanceamento e custos de negocio (para o threshold otimo) vivem em
`config/config.yaml` -- e a unica fonte de verdade, sem duplicacao entre
modulos.

### 4. Notebooks

Os notebooks em `notebooks/` sao material exploratorio (nao fazem parte do
pipeline de producao em `src/fraud_case/`):

- `01_exploratory_analysis.ipynb`
- `02_feature_engineering.ipynb`
- `03_model_evaluation.ipynb`
- `04_hyperparameter_optimization_and_walk_forward.ipynb`

## Decisoes de modelagem

- **Sem vazamento temporal**: toda feature agregada por cartao (media/
  desvio-padrao de gasto, contagem de transacoes em 24h) usa apenas o
  historico estritamente anterior a cada transacao (`shift(1)` +
  `expanding`/`rolling(closed='left')`).
- **Encoding**: `gender` via one-hot; `job`, `city`, `state`, `category`,
  `merchant` via target encoding suavizado (fit somente no treino) --
  `category` e `merchant` sao historicamente os preditores mais fortes
  deste dataset e nao entravam no modelo na versao anterior.
- **Desbalanceamento**: uma unica estrategia por vez, configuravel
  (`class_weight` ou `undersample`), nunca as duas simultaneamente.
- **Metrica principal**: PR-AUC / average precision (ROC-AUC infla a
  percepcao de qualidade com prevalencia ~0.5%).
- **Threshold**: otimizado por custo de negocio (custo de fraude nao
  detectada = valor da transacao; custo de falso positivo = fricao de
  analise manual), nao apenas 0.5 fixo.

## Modelos

Regressao Logistica, Random Forest, Gradient Boosted Trees, XGBoost e
LightGBM, avaliados com AUC-ROC, PR-AUC/average precision, recall a FPR
fixo, precisao, recall, F1 e matriz de confusao.

## Results (test set)

Evaluated on the held-out test set (555,719 transactions, ~0.39% fraud
rate) after the leakage fixes described above — never used to fit the
feature pipeline or train any model:

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **LightGBM** | **0.502** | 0.984 | 0.633 | 0.274 | 0.383 |
| Logistic Regression | 0.303 | 0.967 | 0.084 | 0.851 | 0.153 |
| Random Forest | 0.286 | 0.974 | 0.692 | 0.122 | 0.207 |
| XGBoost | 0.228 | 0.958 | 0.256 | 0.181 | 0.212 |
| Gradient Boosted Trees | 0.218 | 0.972 | 0.147 | 0.325 | 0.147 |

**LightGBM** is the best model by PR-AUC — the metric that matters here,
since with ~0.4% fraud prevalence ROC-AUC alone overstates quality for
every model in the table.

With the business-cost-optimized threshold (0.010, instead of a fixed
0.5) — false-negative cost = transaction amount, false-positive cost =
fixed manual-review cost from `config.yaml` — the estimated total cost on
the test set drops from **$778,286** (at threshold 0.5) to **$210,643**,
an estimated **$567,644 savings**.

Full artifacts (confusion matrices, ROC/PR curves, per-model feature
importance, `model_evaluation_results.csv`, `best_model_summary.txt`) are
generated in `reports/` by `scripts/evaluate_models.py`.

## Licenca

MIT.
