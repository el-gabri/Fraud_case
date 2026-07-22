"""Teste de regressao para o vazamento de dados corrigido em engineering.py.

A versao anterior calculava avg_amt/stddev_amt/amt_zscore com a media e o
desvio-padrao de TODO o historico do cartao (passado E futuro), entao uma
fraude ocorrendo na ultima transacao do cartao influenciava a feature da
PRIMEIRA transacao do mesmo cartao. Isso e logicamente impossivel em
producao (o modelo nao pode ver o futuro) e infla as metricas de avaliacao.

Este teste verifica que as features calculadas para uma transacao antiga
nao mudam quando se altera uma transacao futura do mesmo cartao.
"""

from __future__ import annotations

import pandas as pd

from fraud_case.features.engineering import feature_engineering


def _make_transaction(idx: int, cc_num: str, timestamp: str, amt: float) -> dict:
    return {
        "trans_date_trans_time": timestamp,
        "cc_num": cc_num,
        "merchant": "fraud_Test Merchant",
        "category": "misc_net",
        "amt": amt,
        "first": "Jane",
        "last": "Doe",
        "gender": "F",
        "street": "123 Main St",
        "city": "Testville",
        "state": "NC",
        "zip": 12345,
        "lat": 35.0 + idx * 0.01,
        "long": -80.0 - idx * 0.01,
        "city_pop": 5000,
        "job": "Engineer",
        "dob": "1990-01-01",
        "trans_num": f"txn_{idx}",
        "unix_time": 1546300800 + idx * 3600,
        "merch_lat": 35.1 + idx * 0.01,
        "merch_long": -80.1 - idx * 0.01,
        "is_fraud": 0,
    }


def _build_card_history(third_amt: float) -> pd.DataFrame:
    rows = [
        _make_transaction(0, "1111", "2020-01-01 08:00:00", amt=10.0),
        _make_transaction(1, "1111", "2020-01-02 08:00:00", amt=20.0),
        _make_transaction(2, "1111", "2020-01-03 08:00:00", amt=third_amt),
    ]
    return pd.DataFrame(rows)


def test_past_features_are_unaffected_by_future_transactions():
    df_baseline = _build_card_history(third_amt=30.0)
    df_altered_future = _build_card_history(third_amt=99999.0)  # "fraude" no futuro

    result_baseline = feature_engineering(df_baseline)
    result_altered = feature_engineering(df_altered_future)

    leakage_prone_cols = [
        "avg_amt_expanding",
        "stddev_amt_expanding",
        "amt_zscore_expanding",
        "amt_is_outlier",
        "txn_count_recent",
        "amt_sum_recent",
    ]

    first_two_baseline = result_baseline[result_baseline["trans_num"].isin(["txn_0", "txn_1"])]
    first_two_altered = result_altered[result_altered["trans_num"].isin(["txn_0", "txn_1"])]

    first_two_baseline = first_two_baseline.sort_values("trans_num").reset_index(drop=True)
    first_two_altered = first_two_altered.sort_values("trans_num").reset_index(drop=True)

    for col in leakage_prone_cols:
        pd.testing.assert_series_equal(
            first_two_baseline[col], first_two_altered[col], check_names=False,
        )


def test_first_transaction_of_a_card_has_no_history():
    df = _build_card_history(third_amt=30.0)
    result = feature_engineering(df).sort_values("trans_num").reset_index(drop=True)

    first_row = result.iloc[0]
    assert pd.isna(first_row["avg_amt_expanding"])
    assert pd.isna(first_row["stddev_amt_expanding"])
    assert first_row["amt_zscore_expanding"] == 0.0
    assert first_row["txn_count_recent"] == 0

    second_row = result.iloc[1]
    assert second_row["avg_amt_expanding"] == 10.0  # media de [10.0] (so a 1a transacao)
