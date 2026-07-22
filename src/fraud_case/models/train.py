"""Treinamento dos modelos de deteccao de fraude.

Corrige a dupla-correcao de desbalanceamento da versao anterior, que fazia
undersampling 1:10 *e* usava `class_weight='balanced'`/`scale_pos_weight`
simultaneamente, distorcendo duas vezes a probabilidade estimada
(plan.md secao 2.1, item 4). Agora ha uma unica estrategia, escolhida via
`config.imbalance.strategy`:

- "class_weight" (padrao/recomendado): usa TODOS os dados e corrige o
  desbalanceamento via peso de classe/`scale_pos_weight`. Nao precisa de
  calibracao adicional para o ranking de probabilidades.
- "undersample": os dados de entrada ja devem vir balanceados
  (ver `fraud_case.data.balance.balance_data`); os modelos sao treinados
  sem peso de classe extra, mas as probabilidades resultantes exigem
  calibracao (ver `models/evaluate.py::calibrate_model`).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from fraud_case.config import Config

logger = logging.getLogger(__name__)


def _use_class_weight(config: Config) -> bool:
    return config.imbalance.strategy == "class_weight"


def _scale_pos_weight(y: np.ndarray) -> float:
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    return n_neg / n_pos if n_pos > 0 else 1.0


def train_logistic_regression(X_train, y_train, config: Config) -> LogisticRegression:
    logger.info("Treinando Regressao Logistica...")
    start = time.time()
    params = dict(config.models.get("logistic_regression", {}))
    model = LogisticRegression(
        max_iter=params.get("max_iter", 1000),
        class_weight="balanced" if _use_class_weight(config) else None,
        random_state=config.seed,
    )
    model.fit(X_train, y_train)
    logger.info("Regressao Logistica treinada em %.2fs", time.time() - start)
    return model


def train_random_forest(X_train, y_train, config: Config) -> RandomForestClassifier:
    logger.info("Treinando Random Forest...")
    start = time.time()
    params = dict(config.models.get("random_forest", {}))
    model = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth"),
        max_features="sqrt",
        class_weight="balanced" if _use_class_weight(config) else None,
        random_state=config.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Random Forest treinado em %.2fs", time.time() - start)
    return model


def train_gradient_boosting(X_train, y_train, config: Config) -> GradientBoostingClassifier:
    logger.info("Treinando Gradient Boosted Trees...")
    start = time.time()
    params = dict(config.models.get("gradient_boosting", {}))
    model = GradientBoostingClassifier(
        n_estimators=params.get("n_estimators", 200),
        learning_rate=params.get("learning_rate", 0.1),
        max_depth=params.get("max_depth", 3),
        random_state=config.seed,
    )
    # GradientBoostingClassifier nao aceita class_weight no construtor;
    # a correcao de desbalanceamento e feita via sample_weight no fit.
    sample_weight = (
        compute_sample_weight("balanced", y_train) if _use_class_weight(config) else None
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    logger.info("Gradient Boosted Trees treinado em %.2fs", time.time() - start)
    return model


def train_xgboost(X_train, y_train, config: Config):
    import xgboost as xgb

    logger.info("Treinando XGBoost...")
    start = time.time()
    params = dict(config.models.get("xgboost", {}))
    model = xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 400),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 6),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        objective="binary:logistic",
        scale_pos_weight=_scale_pos_weight(np.asarray(y_train)) if _use_class_weight(config) else 1.0,
        random_state=config.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("XGBoost treinado em %.2fs", time.time() - start)
    return model


def train_lightgbm(X_train, y_train, config: Config):
    import lightgbm as lgb

    logger.info("Treinando LightGBM...")
    start = time.time()
    params = dict(config.models.get("lightgbm", {}))
    model = lgb.LGBMClassifier(
        n_estimators=params.get("n_estimators", 400),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 63),
        class_weight="balanced" if _use_class_weight(config) else None,
        random_state=config.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("LightGBM treinado em %.2fs", time.time() - start)
    return model


TRAINERS = {
    "Logistic Regression": train_logistic_regression,
    "Random Forest": train_random_forest,
    "Gradient Boosted Trees": train_gradient_boosting,
    "XGBoost": train_xgboost,
    "LightGBM": train_lightgbm,
}


def train_all_models(
    X_train: pd.DataFrame, y_train: np.ndarray, config: Config, model_names: list[str] | None = None
) -> dict[str, object]:
    """Treina todos os modelos configurados (ou um subconjunto de `model_names`)."""
    names = model_names or list(TRAINERS.keys())
    models = {}
    for name in names:
        if name not in TRAINERS:
            raise ValueError(f"Modelo desconhecido: {name}. Opcoes: {list(TRAINERS)}")
        models[name] = TRAINERS[name](X_train, y_train, config)
    return models
