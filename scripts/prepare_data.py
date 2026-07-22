#!/usr/bin/env python
"""CLI: carrega os dados brutos, aplica feature engineering (sem vazamento)
e salva os resultados em Parquet em data/processed/.

Uso:
    python scripts/prepare_data.py [--config config/config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_case.config import load_config
from fraud_case.data.balance import balance_data
from fraud_case.data.loader import check_missing_values, load_data
from fraud_case.features.engineering import feature_engineering
from fraud_case.utils.io import save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Caminho para config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if not config.paths.train_path.exists() or not config.paths.test_path.exists():
        logger.error(
            "Arquivos de dados nao encontrados em %s. Coloque fraudTrain.csv e "
            "fraudTest.csv em data/raw/.", config.paths.raw_dir,
        )
        sys.exit(1)

    train_df, test_df = load_data(config.paths.train_path, config.paths.test_path)
    check_missing_values(train_df, test_df)

    short_window = config.velocity_windows["short_window"]

    logger.info("Aplicando feature engineering no treino...")
    train_features = feature_engineering(train_df, short_window=short_window)
    logger.info("Aplicando feature engineering no teste...")
    test_features = feature_engineering(test_df, short_window=short_window)

    if config.imbalance.strategy == "undersample":
        train_features = balance_data(
            train_features,
            target_col=config.columns.target,
            ratio=config.imbalance.undersample_ratio,
            seed=config.seed,
        )

    save_parquet(train_features, config.paths.processed_dir / "train_features.parquet")
    save_parquet(test_features, config.paths.processed_dir / "test_features.parquet")

    logger.info("Preparacao de dados concluida.")


if __name__ == "__main__":
    main()
