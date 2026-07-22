#!/usr/bin/env python
"""Orquestra o pipeline completo: preparacao de dados -> treino -> avaliacao.

Equivalente a rodar em sequencia:
    python scripts/prepare_data.py
    python scripts/train_models.py
    python scripts/evaluate_models.py

As funcoes de EDA ad-hoc que existiam aqui (caract_df, barplot, time_plot,
iqr, numeric_plot) foram removidas: duplicavam funcoes ja existentes em
notebooks/utils.py e nao pertencem ao orquestrador de producao (ver plan.md
secao 3.1). Para analise exploratoria, use os notebooks em notebooks/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scripts.evaluate_models as evaluate_models_cli
import scripts.prepare_data as prepare_data_cli
import scripts.train_models as train_models_cli

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("ETAPA 1/3: Preparacao dos dados")
    prepare_data_cli.main()

    logger.info("ETAPA 2/3: Treinamento dos modelos")
    train_models_cli.main()

    logger.info("ETAPA 3/3: Avaliacao dos modelos")
    evaluate_models_cli.main()


if __name__ == "__main__":
    main()
