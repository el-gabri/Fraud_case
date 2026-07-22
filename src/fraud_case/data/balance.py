"""Undersampling opcional da classe majoritaria.

So deve ser usado quando `imbalance.strategy == 'undersample'` no config.
Nesse caso, a calibracao de probabilidade (ver models/evaluate.py) e
obrigatoria, pois o undersampling distorce a proporcao de classes que o
modelo ve durante o treino (plan.md secao 2.1, item 4).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def balance_data(
    df: pd.DataFrame, target_col: str = "is_fraud", ratio: int = 10, seed: int = 42
) -> pd.DataFrame:
    """Undersampling da classe majoritaria para a proporcao `ratio`:1."""
    fraud_df = df[df[target_col] == 1]
    non_fraud_df = df[df[target_col] == 0]

    fraud_count = len(fraud_df)
    non_fraud_count = len(non_fraud_df)
    sample_size = min(fraud_count * ratio, non_fraud_count)

    non_fraud_sample = non_fraud_df.sample(n=sample_size, random_state=seed)
    balanced_df = pd.concat([fraud_df, non_fraud_sample]).sample(frac=1, random_state=seed)

    logger.info(
        "Balanceamento: %d fraudes, %d nao-fraudes (proporcao original 1:%.1f -> 1:%.1f)",
        fraud_count, len(non_fraud_sample),
        non_fraud_count / fraud_count if fraud_count else float("nan"),
        len(non_fraud_sample) / fraud_count if fraud_count else float("nan"),
    )
    return balanced_df.reset_index(drop=True)
