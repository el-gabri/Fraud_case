#!/usr/bin/env python
"""CLI: avalia os modelos treinados no conjunto de teste (nunca usado no
fit do pipeline nem no treino), com metricas honestas para desbalanceamento
e threshold otimizado por custo de negocio.

Uso:
    python scripts/evaluate_models.py [--config config/config.yaml] [--models XGBoost LightGBM ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib

from fraud_case.config import load_config
from fraud_case.models.evaluate import (
    evaluate_model,
    find_optimal_threshold_by_cost,
    load_model,
    results_to_dataframe,
)
from fraud_case.models.train import TRAINERS
from fraud_case.utils.io import ensure_dir, load_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Caminho para config.yaml")
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Subconjunto de modelos a avaliar (opcoes: {list(TRAINERS)}). Padrao: todos.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    test_path = config.paths.processed_dir / "test_features.parquet"
    if not test_path.exists():
        logger.error("%s nao encontrado. Rode scripts/prepare_data.py primeiro.", test_path)
        sys.exit(1)

    pipeline_path = config.paths.models_dir / "feature_pipeline.joblib"
    if not pipeline_path.exists():
        logger.error("%s nao encontrado. Rode scripts/train_models.py primeiro.", pipeline_path)
        sys.exit(1)

    test_df = load_parquet(test_path)
    feature_cols = config.columns.all_feature_columns
    X_test = test_df[feature_cols]
    y_test = test_df[config.columns.target].to_numpy()
    amt = test_df["amt"].to_numpy()

    pipeline = joblib.load(pipeline_path)
    X_test_transformed = pipeline.transform(X_test)
    feature_names = list(pipeline.get_feature_names_out())

    ensure_dir(config.paths.reports_dir)

    names = args.models or list(TRAINERS.keys())
    results = []
    probabilities = {}
    for name in names:
        model_path = config.paths.models_dir / f"{name.lower().replace(' ', '_')}.joblib"
        if not model_path.exists():
            logger.warning("Modelo '%s' nao encontrado em %s, pulando.", name, model_path)
            continue

        model = load_model(name, config.paths.models_dir)
        model_results = evaluate_model(
            model, X_test_transformed, y_test, name,
            feature_cols=feature_names, output_dir=config.paths.reports_dir,
        )
        results.append(model_results)
        probabilities[name] = model.predict_proba(X_test_transformed)[:, 1]

    if not results:
        logger.error("Nenhum modelo avaliado. Rode scripts/train_models.py primeiro.")
        sys.exit(1)

    results_df = results_to_dataframe(results)
    results_path = config.paths.reports_dir / "model_evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Resultados salvos em: %s", results_path)

    best_idx = results_df["pr_auc"].idxmax()
    best_model_name = results_df.loc[best_idx, "model"]
    logger.info("Melhor modelo (por PR-AUC): %s", best_model_name)

    threshold_result = find_optimal_threshold_by_cost(
        y_test, probabilities[best_model_name], amt,
        cost_fp=config.evaluation.cost_false_positive,
    )
    logger.info(
        "Threshold otimo por custo para %s: %.3f (custo=%.2f, custo@0.5=%.2f, economia=%.2f)",
        best_model_name,
        threshold_result["optimal_threshold"],
        threshold_result["optimal_cost"],
        threshold_result["default_threshold_cost"],
        threshold_result["estimated_savings"],
    )

    summary_path = config.paths.reports_dir / "best_model_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        row = results_df.loc[best_idx]
        f.write("RESUMO DO MODELO DE DETECCAO DE FRAUDE\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Melhor modelo (PR-AUC): {best_model_name}\n")
        f.write(f"PR-AUC (average precision): {row['average_precision']:.4f}\n")
        f.write(f"ROC-AUC: {row['roc_auc']:.4f}\n")
        f.write(f"Precisao: {row['precision']:.4f}\n")
        f.write(f"Recall: {row['recall']:.4f}\n")
        f.write(f"F1-Score: {row['f1']:.4f}\n\n")
        f.write(
            f"Threshold otimo por custo de negocio: {threshold_result['optimal_threshold']:.3f}\n"
        )
        f.write(f"Custo estimado no threshold otimo: {threshold_result['optimal_cost']:.2f}\n")
        f.write(
            f"Custo estimado no threshold padrao (0.5): "
            f"{threshold_result['default_threshold_cost']:.2f}\n"
        )
        f.write(f"Economia estimada: {threshold_result['estimated_savings']:.2f}\n")
    logger.info("Resumo salvo em: %s", summary_path)


if __name__ == "__main__":
    main()
