"""Features derivadas de data/hora, calculadas point-in-time.

O bug corrigido aqui (ver plan.md secao 2.1, item 5): a versao anterior
calculava a idade usando uma data de referencia FIXA (2023-01-01), mesmo
para transacoes de 2019. A idade deve ser calculada na data da PROPRIA
transacao, senao o modelo aprende uma idade errada e nao generaliza para
novos periodos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def age_at_date(dob: pd.Series, reference_date: pd.Series) -> pd.Series:
    """Idade (em anos completos) de `dob` na data `reference_date`.

    Ambas devem ser Series de datetime64 (ou conversíveis), do mesmo tamanho.
    """
    dob = pd.to_datetime(dob)
    reference_date = pd.to_datetime(reference_date)

    years = reference_date.dt.year - dob.dt.year
    had_birthday = (reference_date.dt.month > dob.dt.month) | (
        (reference_date.dt.month == dob.dt.month) & (reference_date.dt.day >= dob.dt.day)
    )
    age = years - (~had_birthday).astype(int)
    return age.astype("float").where(dob.notna() & reference_date.notna())


def cyclical_encode(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """Codifica uma variavel ciclica (hora, dia da semana...) em seno/cosseno.

    Evita que o modelo trate, por exemplo, 23h e 0h como extremos opostos.
    """
    radians = 2 * np.pi * values.astype(float) / period
    return np.sin(radians), np.cos(radians)
