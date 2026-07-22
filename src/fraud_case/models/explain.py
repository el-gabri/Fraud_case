"""Explicabilidade do modelo final via SHAP (plan.md secao 4.5).

Import do shap e feito dentro da funcao (dependencia pesada e opcional;
so e necessaria na fase final de explicabilidade, nao no treino/avaliacao).
"""

from __future__ import annotations

import os

import pandas as pd


def compute_shap_values(model, X_sample: pd.DataFrame):
    """Calcula valores SHAP para uma amostra (recomendado: poucas centenas
    de linhas, TreeExplainer e rapido mas o plot fica ilegivel com muitas).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    return explainer(X_sample)


def save_summary_plot(shap_values, X_sample: pd.DataFrame, save_path: str | os.PathLike) -> None:
    import matplotlib

    # Backend nao-interativo (ver utils/plotting.py): so salvamos em
    # arquivo, e o backend interativo padrao quebra sem um main loop de GUI.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
