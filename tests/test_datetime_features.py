import numpy as np
import pandas as pd

from fraud_case.utils.datetime_features import age_at_date, cyclical_encode


def test_age_before_birthday_this_year():
    dob = pd.Series(pd.to_datetime(["1990-06-15"]))
    reference = pd.Series(pd.to_datetime(["2020-06-14"]))  # um dia antes do aniversario
    age = age_at_date(dob, reference)
    assert age.iloc[0] == 29


def test_age_on_or_after_birthday_this_year():
    dob = pd.Series(pd.to_datetime(["1990-06-15"]))
    reference = pd.Series(pd.to_datetime(["2020-06-15"]))  # no dia do aniversario
    age = age_at_date(dob, reference)
    assert age.iloc[0] == 30


def test_age_uses_transaction_date_not_fixed_reference():
    """Regressao do bug: a versao antiga usava uma data fixa (2023-01-01)
    mesmo para transacoes de 2019, inflando a idade em anos."""
    dob = pd.Series(pd.to_datetime(["1988-03-09", "1988-03-09"]))
    references = pd.Series(pd.to_datetime(["2019-01-01", "2023-01-01"]))
    ages = age_at_date(dob, references)
    assert ages.iloc[0] == 30  # na data da transacao (2019), nao 34/35 (2023)
    assert ages.iloc[1] == 34


def test_cyclical_encode_wraps_around():
    hours = pd.Series([0, 23])
    sin_vals, cos_vals = cyclical_encode(hours, period=24)
    # 0h e 23h devem ficar proximas no espaco seno/cosseno (diferenca de 1h),
    # nao nos extremos opostos como aconteceria com a hora crua.
    dist = np.hypot(sin_vals.iloc[0] - sin_vals.iloc[1], cos_vals.iloc[0] - cos_vals.iloc[1])
    assert dist < 0.3
