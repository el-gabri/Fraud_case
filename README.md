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

## Licenca

MIT.
