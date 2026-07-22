#!/usr/bin/env python
"""CLI: ajusta o pipeline de features (fit somente no treino) e treina os
modelos de deteccao de fraude.

Uso:
    python scripts/train_models.py [--config config/config.yaml] [--models XGBoost LightGBM ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib

from fraud_case.config import load_config
from fraud_case.features.pipeline import build_feature_pipeline
from fraud_case.models.evaluate import save_model
from fraud_case.models.train import TRAINERS, train_all_models
from fraud_case.utils.io import ensure_dir, load_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Caminho para config.yaml")
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Subconjunto de modelos a treinar (opcoes: {list(TRAINERS)}). Padrao: todos.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    train_path = config.paths.processed_dir / "train_features.parquet"
    if not train_path.exists():
        logger.error("%s nao encontrado. Rode scripts/prepare_data.py primeiro.", train_path)
        sys.exit(1)

    train_df = load_parquet(train_path)
    feature_cols = config.columns.all_feature_columns
    X_train = train_df[feature_cols]
    y_train = train_df[config.columns.target].to_numpy()

    logger.info("Ajustando pipeline de features no treino (%d linhas)...", len(train_df))
    pipeline = build_feature_pipeline(config.columns, config.target_encoding)
    X_train_transformed = pipeline.fit_transform(X_train, y_train)

    ensure_dir(config.paths.models_dir)
    pipeline_path = config.paths.models_dir / "feature_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    logger.info("Pipeline de features salvo em: %s", pipeline_path)

    models = train_all_models(X_train_transformed, y_train, config, model_names=args.models)
    for name, model in models.items():
        save_model(model, name, config.paths.models_dir)

    logger.info("Treinamento concluido: %s", list(models))


if __name__ == "__main__":
    main()
