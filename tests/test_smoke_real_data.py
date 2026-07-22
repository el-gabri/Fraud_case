"""Smoke test com uma amostra real de data/raw/fraudTrain.csv.

Verifica que o pipeline de feature engineering roda ponta a ponta em dados
reais (nao so em fixtures sinteticas) e produz exatamente as colunas que
config/config.yaml espera. Pulado automaticamente se o CSV bruto nao
estiver presente (ele nao e versionado no git - ver .gitignore).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fraud_case.config import load_config
from fraud_case.features.engineering import feature_engineering
from fraud_case.utils.io import read_transactions_csv

RAW_TRAIN_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "fraudTrain.csv"

pytestmark = pytest.mark.skipif(
    not RAW_TRAIN_CSV.exists(), reason="data/raw/fraudTrain.csv nao encontrado (nao versionado)"
)


@pytest.fixture(scope="module")
def sample_df():
    # nrows suficiente para conter cartoes (cc_num) repetidos e exercitar
    # de fato as features de expanding/rolling window.
    return read_transactions_csv(RAW_TRAIN_CSV, nrows=5000)


def test_feature_engineering_runs_on_real_sample(sample_df):
    result = feature_engineering(sample_df)
    assert len(result) == len(sample_df)


def test_all_configured_feature_columns_are_present(sample_df):
    config = load_config()
    result = feature_engineering(sample_df)

    missing = [c for c in config.columns.all_feature_columns if c not in result.columns]
    assert not missing, f"Colunas esperadas pelo config.yaml ausentes: {missing}"


def test_no_unexpected_nans_in_always_computable_columns(sample_df):
    result = feature_engineering(sample_df)

    # Estas nao dependem de historico do cartao, entao nunca devem ser NaN.
    always_present = [
        "distance_km", "first_digit", "last_digit", "amt_log", "city_pop_log",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month",
    ]
    for col in always_present:
        assert result[col].isna().sum() == 0, f"Coluna {col} tem NaN inesperado"


def test_repeated_card_has_some_transactions_with_history(sample_df):
    """Confirma que a amostra de fato contem cartoes repetidos, senao os
    testes de expanding-window acima estariam vacuamente satisfeitos."""
    result = feature_engineering(sample_df)
    assert result["avg_amt_expanding"].notna().sum() > 0
