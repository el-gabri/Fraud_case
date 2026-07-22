"""Carregamento dos dados brutos de transacoes."""

from __future__ import annotations

import logging
import os

import pandas as pd

from fraud_case.utils.io import read_transactions_csv

logger = logging.getLogger(__name__)


def load_data(train_path: str | os.PathLike, test_path: str | os.PathLike) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os conjuntos de treino e teste com dtypes explicitos."""
    logger.info("Carregando dados de treino e teste...")
    train_df = read_transactions_csv(train_path)
    test_df = read_transactions_csv(test_path)

    logger.info("Linhas no treino: %d | Linhas no teste: %d", len(train_df), len(test_df))
    return train_df, test_df


def check_missing_values(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Retorna (e loga) a contagem de valores ausentes por coluna em cada conjunto."""
    train_missing = train_df.isnull().sum()
    train_missing = train_missing[train_missing > 0]

    test_missing = test_df.isnull().sum()
    test_missing = test_missing[test_missing > 0]

    if len(train_missing) > 0:
        logger.info("Valores ausentes no treino:\n%s", train_missing)
    else:
        logger.info("Nao ha valores ausentes no treino.")

    if len(test_missing) > 0:
        logger.info("Valores ausentes no teste:\n%s", test_missing)
    else:
        logger.info("Nao ha valores ausentes no teste.")

    return train_missing, test_missing
