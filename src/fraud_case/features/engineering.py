"""Engenharia de features -- reescrita para eliminar vazamento de dados.

Problemas corrigidos em relacao a versao anterior (ver plan.md secao 2.1):

1. `avg_amt`/`stddev_amt`/`amt_zscore` eram calculados com a media/desvio de
   TODO o historico do cartao (passado E futuro). Agora usam `shift(1)` +
   `expanding()`, enxergando apenas transacoes estritamente anteriores.
2. `first_digit`/`last_digit` tinham um bug de regex (o `str.replace` era
   um no-op) que zerava o primeiro digito para valores < 1. Corrigidos com
   aritmetica (log10 / centavos inteiros).
3. `idade` usava uma data de referencia fixa (2023-01-01) mesmo para
   transacoes de 2019. Agora calculada na data da PROPRIA transacao.
4. `unix_time`/`prev_unix_time`/coordenadas absolutas (`lat`, `long`,
   `merch_lat`, `merch_long`, `zip`) nao entram mais como features de
   modelo -- sao apenas insumos intermediarios para calcular distancias e
   diferencas de tempo, que sao relativas e generalizam melhor.

Alem disso, adiciona features de velocity de curto prazo (contagem/soma de
transacoes nas ultimas 24h por cartao), que sao historicamente fortes
preditores de fraude e nao existiam na versao anterior (plan.md secao 4.2).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fraud_case.utils.datetime_features import age_at_date, cyclical_encode
from fraud_case.utils.geo import haversine_distance

logger = logging.getLogger(__name__)

_EPS = 1e-10


def _leading_digit(amt: pd.Series) -> pd.Series:
    """Primeiro digito significativo do valor (ex.: 45.67 -> 4, 0.99 -> 9)."""
    amt = amt.astype(float)
    positive = amt > 0
    safe_amt = amt.where(positive, 1.0)
    exponent = np.floor(np.log10(safe_amt) + _EPS)
    leading = (safe_amt / (10.0 ** exponent)).astype(int)
    return leading.where(positive, 0).astype(int)


def _last_cents_digit(amt: pd.Series) -> pd.Series:
    """Ultimo digito dos centavos (ex.: 45.67 -> 7). Baseado em centavos
    inteiros para nao depender de quantas casas decimais o float imprime.
    """
    cents = np.round(amt.astype(float) * 100).astype("int64")
    return (cents % 10).astype(int)


def feature_engineering(df: pd.DataFrame, short_window: str = "24h") -> pd.DataFrame:
    """Aplica toda a engenharia de features, sem vazamento temporal.

    Pressupoe que `df` tem as colunas brutas do dataset (fraudTrain/Test.csv)
    com `trans_date_trans_time` e `dob` ja convertidos para datetime
    (ver `fraud_case.utils.io.read_transactions_csv`).

    `short_window` controla a janela de velocity de curto prazo
    (`txn_count_recent`/`amt_sum_recent`); configuravel via
    `config.yaml: velocity_windows.short_window`.
    """
    logger.info("Iniciando feature engineering...")
    df_new = df.copy()

    df_new["trans_date_trans_time"] = pd.to_datetime(df_new["trans_date_trans_time"])
    df_new["dob"] = pd.to_datetime(df_new["dob"])

    # 1. Componentes temporais (+ codificacao ciclica: 23h e 0h sao vizinhas)
    df_new["hour_of_day"] = df_new["trans_date_trans_time"].dt.hour
    df_new["day_of_week"] = df_new["trans_date_trans_time"].dt.dayofweek + 1  # 1=Seg, 7=Dom
    df_new["month"] = df_new["trans_date_trans_time"].dt.month

    df_new["hour_sin"], df_new["hour_cos"] = cyclical_encode(df_new["hour_of_day"], period=24)
    df_new["dow_sin"], df_new["dow_cos"] = cyclical_encode(df_new["day_of_week"], period=7)

    # 2. Distancia cliente-comerciante
    df_new["distance_km"] = haversine_distance(
        df_new["lat"].values, df_new["long"].values,
        df_new["merch_lat"].values, df_new["merch_long"].values,
    )

    # 3. Digitos do valor da transacao (analise tipo Lei de Benford)
    df_new["first_digit"] = _leading_digit(df_new["amt"])
    df_new["last_digit"] = _last_cents_digit(df_new["amt"])

    # 4. Log-transform de valores com cauda longa
    df_new["amt_log"] = np.log1p(df_new["amt"].clip(lower=0))
    df_new["city_pop_log"] = np.log1p(df_new["city_pop"].clip(lower=0))

    # 5. Idade na data da transacao (point-in-time; corrige o bug da data fixa)
    df_new["idade"] = age_at_date(df_new["dob"], df_new["trans_date_trans_time"])

    # A partir daqui, todas as features dependem da ordem cronologica por
    # cartao: qualquer estatistica agregada so pode olhar para o PASSADO.
    df_new = df_new.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)

    # 6. Distancia/velocidade em relacao a transacao anterior do mesmo cartao
    grouped = df_new.groupby("cc_num", sort=False)
    prev_lat = grouped["lat"].shift(1)
    prev_long = grouped["long"].shift(1)
    prev_timestamp = grouped["trans_date_trans_time"].shift(1)

    df_new["time_diff_hours"] = (
        df_new["trans_date_trans_time"] - prev_timestamp
    ).dt.total_seconds() / 3600

    has_prev = prev_lat.notna() & prev_long.notna()
    prev_distance_km = pd.Series(np.nan, index=df_new.index)
    prev_distance_km.loc[has_prev] = haversine_distance(
        df_new.loc[has_prev, "lat"].values,
        df_new.loc[has_prev, "long"].values,
        prev_lat.loc[has_prev].values,
        prev_long.loc[has_prev].values,
    )

    valid_velocity = has_prev & (df_new["time_diff_hours"] > 0)
    df_new["transaction_velocity_kmh"] = np.where(
        valid_velocity, prev_distance_km / df_new["time_diff_hours"], np.nan,
    )
    df_new["impossible_velocity"] = np.where(df_new["transaction_velocity_kmh"] > 1000, 1, 0)

    # 7. Media/desvio-padrao de gasto do cartao ATE a transacao anterior
    #    (shift(1) antes do expanding: a linha atual nunca entra na propria
    #    estatistica, eliminando o vazamento da versao original).
    amt_by_card = df_new.groupby("cc_num", sort=False)["amt"]
    df_new["avg_amt_expanding"] = amt_by_card.transform(lambda s: s.shift(1).expanding().mean())
    df_new["stddev_amt_expanding"] = amt_by_card.transform(lambda s: s.shift(1).expanding().std())

    has_history = df_new["stddev_amt_expanding"].notna() & (df_new["stddev_amt_expanding"] > 0)
    df_new["amt_zscore_expanding"] = np.where(
        has_history,
        (df_new["amt"] - df_new["avg_amt_expanding"]) / df_new["stddev_amt_expanding"],
        0.0,
    )
    df_new["amt_is_outlier"] = np.where(np.abs(df_new["amt_zscore_expanding"]) > 3, 1, 0)

    # 8. Velocity de curto prazo: nº e soma de transacoes do cartao na
    #    janela `short_window`, olhando estritamente para o passado
    #    (closed='left' exclui a propria transacao da janela).
    time_indexed = df_new[["cc_num", "trans_date_trans_time", "amt"]].set_index(
        "trans_date_trans_time"
    )
    rolling_recent = time_indexed.groupby("cc_num", sort=False)["amt"].rolling(
        short_window, closed="left"
    )
    df_new["txn_count_recent"] = (
        rolling_recent.count().reset_index(level=0, drop=True).fillna(0).to_numpy()
    )
    df_new["amt_sum_recent"] = (
        rolling_recent.sum().reset_index(level=0, drop=True).fillna(0).to_numpy()
    )

    logger.info("Feature engineering concluida: %d colunas.", df_new.shape[1])
    return df_new
