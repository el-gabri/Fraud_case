"""Utilidades de I/O (diretorios, leitura/escrita de dados)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def ensure_dir(directory: str | os.PathLike) -> Path:
    """Garante que o diretorio existe, criando-o se necessario."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_transactions_csv(path: str | os.PathLike, **read_csv_kwargs) -> pd.DataFrame:
    """Le o CSV bruto de transacoes com dtypes explicitos.

    `cc_num` e um identificador (nao uma quantidade), entao e lido como
    string para evitar problemas de precisao/formatacao numerica.

    Aceita kwargs extras de `pandas.read_csv` (ex.: `nrows=` para ler uma
    amostra rapida em testes).
    """
    return pd.read_csv(
        path,
        dtype={
            "cc_num": "string",
            "merchant": "category",
            "category": "category",
            "gender": "category",
            "job": "category",
            "city": "category",
            "state": "category",
        },
        parse_dates=["trans_date_trans_time", "dob"],
        **read_csv_kwargs,
    )


def save_parquet(df: pd.DataFrame, path: str | os.PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    logger.info("Dados salvos em %s (%d linhas)", path, len(df))


def load_parquet(path: str | os.PathLike) -> pd.DataFrame:
    return pd.read_parquet(path)
