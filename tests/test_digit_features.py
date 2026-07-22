import pandas as pd

from fraud_case.features.engineering import _last_cents_digit, _leading_digit


def test_leading_digit_for_values_below_one():
    """Regressao do bug: `str.replace(r'^0+', '')` sem regex=True era um
    no-op, entao amt < 1 (ex.: 0.99) tinha 'first_digit' = 0 em vez de 9."""
    amounts = pd.Series([0.99, 0.05, 0.5])
    result = _leading_digit(amounts)
    assert list(result) == [9, 5, 5]


def test_leading_digit_for_values_above_one():
    amounts = pd.Series([45.67, 100.0, 999.99, 1.0])
    result = _leading_digit(amounts)
    assert list(result) == [4, 1, 9, 1]


def test_leading_digit_zero_is_zero():
    amounts = pd.Series([0.0])
    result = _leading_digit(amounts)
    assert list(result) == [0]


def test_last_cents_digit_is_consistent_regardless_of_decimal_places():
    """Regressao do bug: usar str(amt) dava semantica inconsistente porque
    o Python omite zeros decimais a direita (45.5 -> '45.5', 45.50 -> mesma
    string). Trabalhando em centavos inteiros o resultado e bem definido."""
    amounts = pd.Series([45.67, 45.5, 45.0, 45.50])
    result = _last_cents_digit(amounts)
    assert list(result) == [7, 0, 0, 0]
