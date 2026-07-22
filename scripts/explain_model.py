#!/usr/bin/env python
"""CLI: gera explicabilidade SHAP para um modelo ja treinado e avaliado
(plan.md secao 4.5) -- essencial para justificar por que o modelo marcou
uma transacao como fraude.

Usa uma amostra do teste (TreeExplainer e rapido, mas o summary_plot fica
ilegivel e lento com centenas de milhares de linhas de uma vez).

Uso:
    python scripts/explain_model.py [--model LightGBM] [--sample-size 2000]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib

from fraud_case.config import load_config
from fraud_case.models.evaluate import load_model
from fraud_case.models.explain import compute_shap_values, save_summary_plot
from fraud_case.utils.io import ensure_dir, load_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Caminho para config.yaml")
    parser.add_argument("--model", default="LightGBM", help="Nome do modelo ja treinado")
    parser.add_argument("--sample-size", type=int, default=2000)
    args = parser.parse_args()

    config = load_config(args.config)

    test_path = config.paths.processed_dir / "test_features.parquet"
    pipeline_path = config.paths.models_dir / "feature_pipeline.joblib"
    if not test_path.exists() or not pipeline_path.exists():
        logger.error(
            "Dados/pipeline nao encontrados. Rode scripts/prepare_data.py e "
            "scripts/train_models.py primeiro."
        )
        sys.exit(1)

    test_df = load_parquet(test_path)
    feature_cols = config.columns.all_feature_columns

    pipeline = joblib.load(pipeline_path)
    model = load_model(args.model, config.paths.models_dir)

    sample_size = min(args.sample_size, len(test_df))
    sample_df = test_df.sample(n=sample_size, random_state=config.seed)
    X_sample = pipeline.transform(sample_df[feature_cols])

    logger.info("Calculando SHAP values para %d linhas (%s)...", sample_size, args.model)
    shap_values = compute_shap_values(model, X_sample)

    ensure_dir(config.paths.reports_dir)
    safe_name = args.model.lower().replace(" ", "_")
    save_path = config.paths.reports_dir / f"{safe_name}_shap_summary.png"
    save_summary_plot(shap_values, X_sample, save_path)
    logger.info("SHAP summary plot salvo em: %s", save_path)


if __name__ == "__main__":
    main()
