"""Avaliacao de modelos: metricas honestas para dados desbalanceados,
threshold otimizado por custo de negocio e calibracao de probabilidade.

Por que PR-AUC/average precision em vez de so ROC-AUC: com prevalencia de
fraude ~0.5%, o ROC-AUC infla a percepcao de qualidade porque a taxa de
falsos positivos (FPR) permanece baixa mesmo com muitos falsos positivos
em termos absolutos (plan.md secao 4.3).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from fraud_case.utils.io import ensure_dir
from fraud_case.utils.plotting import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
    feature_cols: list[str] | None = None,
    output_dir: str | os.PathLike | None = None,
    target_fpr: float = 0.001,
) -> dict:
    """Avalia o modelo e (opcionalmente) salva graficos/artefatos em `output_dir`."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    prevalence = float(np.mean(y_test))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": pr_auc,
        "average_precision": average_precision_score(y_test, y_prob),
        f"recall_at_fpr_{target_fpr}": float(np.interp(target_fpr, fpr, tpr)),
        "prevalence": prevalence,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

    logger.info("Avaliacao de %s: %s", model_name, results)

    if output_dir:
        ensure_dir(output_dir)
        safe_name = model_name.lower().replace(" ", "_")

        plot_confusion_matrix(
            cm, title=f"Matriz de Confusao - {model_name}",
            save_path=os.path.join(output_dir, f"{safe_name}_confusion_matrix.png"),
        )
        plot_roc_curve(
            fpr, tpr, results["roc_auc"], title=f"Curva ROC - {model_name}",
            save_path=os.path.join(output_dir, f"{safe_name}_roc_curve.png"),
        )
        plot_precision_recall_curve(
            recall_curve, precision_curve, pr_auc, prevalence=prevalence,
            title=f"Curva Precision-Recall - {model_name}",
            save_path=os.path.join(output_dir, f"{safe_name}_pr_curve.png"),
        )

        if hasattr(model, "feature_importances_") and feature_cols:
            importance_df = plot_feature_importance(
                feature_cols, model.feature_importances_,
                title=f"Importancia das Features - {model_name}",
                save_path=os.path.join(output_dir, f"{safe_name}_feature_importance.png"),
            )
            importance_df.to_csv(
                os.path.join(output_dir, f"{safe_name}_feature_importance.csv"), index=False
            )

    return results


def find_optimal_threshold_by_cost(
    y_true, y_prob, amt, cost_fp: float = 5.0, thresholds: np.ndarray | None = None
) -> dict:
    """Encontra o threshold que minimiza o custo de negocio esperado.

    Custo de falso negativo (fraude nao detectada) = valor da transacao
    (`amt`); custo de falso positivo (fricção/analise manual) = `cost_fp`
    fixo. Substitui `util.py::plot_threshold_analysis` (que so plotava,
    sem separar a logica de calculo) -- ver plan.md secao 4.3.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    amt = np.asarray(amt, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    costs = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        y_pred = (y_prob >= t).astype(int)
        fn_mask = (y_true == 1) & (y_pred == 0)
        fp_mask = (y_true == 0) & (y_pred == 1)
        costs[i] = amt[fn_mask].sum() + cost_fp * fp_mask.sum()

    best_idx = int(np.argmin(costs))

    default_pred = (y_prob >= 0.5).astype(int)
    default_fn_mask = (y_true == 1) & (default_pred == 0)
    default_fp_mask = (y_true == 0) & (default_pred == 1)
    default_cost = float(amt[default_fn_mask].sum() + cost_fp * default_fp_mask.sum())

    return {
        "optimal_threshold": float(thresholds[best_idx]),
        "optimal_cost": float(costs[best_idx]),
        "default_threshold_cost": default_cost,
        "estimated_savings": default_cost - float(costs[best_idx]),
        "thresholds": thresholds,
        "costs": costs,
    }


def calibrate_model(model, X_calib, y_calib, method: str = "isotonic"):
    """Calibra as probabilidades de um modelo ja treinado (obrigatorio se
    `imbalance.strategy == 'undersample'`, ver train.py).
    """
    from sklearn.calibration import CalibratedClassifierCV

    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    except ImportError:
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")

    calibrated.fit(X_calib, y_calib)
    return calibrated


def save_model(model, model_name: str, models_dir: str | os.PathLike) -> Path:
    models_dir = ensure_dir(models_dir)
    path = Path(models_dir) / f"{model_name.lower().replace(' ', '_')}.joblib"
    joblib.dump(model, path)
    logger.info("Modelo '%s' salvo em: %s", model_name, path)
    return path


def load_model(model_name: str, models_dir: str | os.PathLike):
    path = Path(models_dir) / f"{model_name.lower().replace(' ', '_')}.joblib"
    return joblib.load(path)


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results)
