"""Visualizacoes de avaliacao de modelo (matriz de confusao, ROC, PR, importancia).

Consolidado a partir do antigo src/utils.py — a versao anterior tinha as
mesmas funcoes duplicadas em src/utils.py, util.py (raiz) e notebooks/utils.py.
"""

from __future__ import annotations

import logging
import os

import matplotlib

# Backend nao-interativo: este modulo so salva figuras em arquivo (savefig),
# nunca chama plt.show(). O backend interativo padrao (TkAgg) quebra quando
# rodado sem um main loop de GUI (ex.: scripts em background/subprocess).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def _maybe_save(save_path: str | os.PathLike | None) -> None:
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Figura salva em: %s", save_path)


def plot_confusion_matrix(cm, title="Matriz de Confusao", cmap=None, save_path=None):
    cmap = cmap or plt.cm.Blues
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=["Nao Fraude", "Fraude"],
        yticklabels=["Nao Fraude", "Fraude"],
    )
    plt.title(title)
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Prevista")
    plt.tight_layout()
    _maybe_save(save_path)
    plt.close()


def plot_feature_importance(feature_names, importances, title="Importancia das Features",
                             save_path=None, top_n=25):
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title(title)
    plt.tight_layout()
    _maybe_save(save_path)
    plt.close()

    return importance_df


def plot_roc_curve(fpr, tpr, roc_auc, title="Curva ROC", save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title(title)
    plt.legend(loc="lower right")
    _maybe_save(save_path)
    plt.close()


def plot_precision_recall_curve(recall, precision, pr_auc, prevalence=None,
                                 title="Curva Precision-Recall", save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2, label=f"AUC-PR = {pr_auc:.3f}")
    if prevalence is not None:
        plt.axhline(prevalence, color="gray", linestyle="--",
                    label=f"Baseline aleatorio ({prevalence:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    _maybe_save(save_path)
    plt.close()
