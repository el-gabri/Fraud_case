"""Pipeline de transformacao de features via sklearn Pipeline/ColumnTransformer.

Substitui o `apply_transformations` manual da versao anterior (que ajustava
o StandardScaler already-undersampled e concatenava DataFrames na mao). Usar
um ColumnTransformer elimina por construcao o risco de "fit no teste":
`pipeline.fit(X_train, y_train)` e depois `pipeline.transform(X_test)`.

Tambem substitui o one-hot encoding de `job`/`city` (centenas de colunas
esparsas) por um target encoder suavizado, que tambem cobre `category`,
`merchant` e `state` -- historicamente os preditores mais fortes deste
dataset e que a versao anterior ignorava (plan.md secao 4.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_case.config import ColumnsConfig, TargetEncodingConfig


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoding com m-estimate smoothing (fit somente no treino).

    Categorias com poucas amostras (< min_samples_leaf) sao puxadas em
    direcao a media global, evitando overfitting em categorias raras.
    Categorias nao vistas no fit (ex.: um `job` novo no teste) recebem a
    media global.
    """

    def __init__(self, columns: list[str], smoothing: float = 20.0, min_samples_leaf: int = 10):
        self.columns = columns
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("SmoothedTargetEncoder requer y no fit.")
        X = self._to_frame(X)
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)

        self.global_mean_ = float(y.mean())
        self.encodings_: dict[str, pd.Series] = {}
        for col in self.columns:
            stats = y.groupby(X[col], observed=True).agg(["mean", "count"])
            weight = 1.0 / (
                1.0 + np.exp(-(stats["count"] - self.min_samples_leaf) / self.smoothing)
            )
            self.encodings_[col] = self.global_mean_ * (1 - weight) + stats["mean"] * weight
        return self

    def transform(self, X):
        X = self._to_frame(X)
        out = pd.DataFrame(index=X.index)
        for col in self.columns:
            mapping = self.encodings_[col]
            out[f"{col}_target_enc"] = (
                X[col].map(mapping).astype(float).fillna(self.global_mean_)
            )
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_target_enc" for c in self.columns])

    def _to_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.columns)


def build_feature_pipeline(
    columns: ColumnsConfig, target_encoding: TargetEncodingConfig
) -> ColumnTransformer:
    """Monta o ColumnTransformer usado por todos os modelos.

    Uso tipico:
        pipeline = build_feature_pipeline(config.columns, config.target_encoding)
        X_train_t = pipeline.fit_transform(X_train, y_train)
        X_test_t = pipeline.transform(X_test)
    """
    target_encoder = SmoothedTargetEncoder(
        columns=columns.categorical_high_cardinality,
        smoothing=target_encoding.smoothing,
        min_samples_leaf=target_encoding.min_samples_leaf,
    )

    # Features de "expanding window" (avg_amt_expanding, time_diff_hours...)
    # sao NaN na primeira transacao de cada cartao (ainda nao ha historico).
    # A imputacao por mediana e ajustada apenas no treino, sem vazamento.
    numeric_scaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("scaled", numeric_scaled, columns.numeric_to_scale),
            ("passthrough", "passthrough", columns.numeric_passthrough),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                columns.categorical_low_cardinality,
            ),
            ("target_enc", target_encoder, columns.categorical_high_cardinality),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    transformer.set_output(transform="pandas")
    return transformer
