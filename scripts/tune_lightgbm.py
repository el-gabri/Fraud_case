#!/usr/bin/env python
"""CLI: ajuste de hiperparametros do LightGBM via Optuna, com validacao
baseada em TEMPO (nao k-fold aleatorio) -- consistente com a natureza
sequencial do problema de fraude (plan.md secao 4.4).

A busca usa um recorte temporal do treino (ultimos `--val-fraction` por
tempo, nunca aleatorio) para nao vazar informacao futura na escolha de
hiperparametros. O pipeline de features usado durante a busca e ajustado
SOMENTE na fatia de tuning-treino (nao no treino completo), para a metrica
de validacao ser honesta.

Depois de escolhidos os melhores hiperparametros, o modelo final e
retreinado com TODO o treino, reaproveitando o pipeline de features ja
ajustado por scripts/train_models.py (models/feature_pipeline.joblib) --
assim fica diretamente comparavel aos demais modelos em
scripts/evaluate_models.py. Salvo como "LightGBM Tuned".

Uso:
    python scripts/tune_lightgbm.py [--config config/config.yaml] [--trials 25] [--val-fraction 0.2]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud_case.config import Config, load_config
from fraud_case.features.pipeline import build_feature_pipeline
from fraud_case.models.evaluate import save_model
from fraud_case.utils.io import ensure_dir, load_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def time_based_split(
    df: pd.DataFrame, val_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ultimas `val_fraction` transacoes por tempo viram validacao.

    Diferente de um split aleatorio/k-fold: replica o cenario real de
    producao, onde o modelo so pode aprender com o passado.
    """
    df_sorted = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - val_fraction))
    return df_sorted.iloc[:split_idx].reset_index(drop=True), df_sorted.iloc[split_idx:].reset_index(drop=True)


def _lgbm_kwargs(params: dict, config: Config) -> dict:
    return dict(
        **params,
        class_weight="balanced" if config.imbalance.strategy == "class_weight" else None,
        random_state=config.seed,
        n_jobs=-1,
        verbosity=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Caminho para config.yaml")
    parser.add_argument("--trials", type=int, default=25, help="Numero de trials do Optuna")
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="Fracao (por tempo) do treino usada como validacao na busca",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    train_path = config.paths.processed_dir / "train_features.parquet"
    if not train_path.exists():
        logger.error("%s nao encontrado. Rode scripts/prepare_data.py primeiro.", train_path)
        sys.exit(1)

    pipeline_path = config.paths.models_dir / "feature_pipeline.joblib"
    if not pipeline_path.exists():
        logger.error("%s nao encontrado. Rode scripts/train_models.py primeiro.", pipeline_path)
        sys.exit(1)

    full_train_df = load_parquet(train_path)
    feature_cols = config.columns.all_feature_columns
    target_col = config.columns.target

    logger.info(
        "Split temporal para tuning: %.0f%% tuning-treino / %.0f%% validacao",
        (1 - args.val_fraction) * 100, args.val_fraction * 100,
    )
    tune_train_df, tune_val_df = time_based_split(full_train_df, args.val_fraction)
    logger.info(
        "Tuning-treino: %d linhas (%.3f%% fraude) | Validacao: %d linhas (%.3f%% fraude)",
        len(tune_train_df), 100 * tune_train_df[target_col].mean(),
        len(tune_val_df), 100 * tune_val_df[target_col].mean(),
    )

    # Pipeline temporario, ajustado SO na fatia de tuning-treino -- separado
    # do pipeline final (que usa todo o treino) para a busca ser honesta.
    tuning_pipeline = build_feature_pipeline(config.columns, config.target_encoding)
    y_tune_train = tune_train_df[target_col].to_numpy()
    y_tune_val = tune_val_df[target_col].to_numpy()
    X_tune_train = tuning_pipeline.fit_transform(tune_train_df[feature_cols], y_tune_train)
    X_tune_val = tuning_pipeline.transform(tune_val_df[feature_cols])

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 1, 7),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        )
        model = lgb.LGBMClassifier(**_lgbm_kwargs(params, config))
        model.fit(X_tune_train, y_tune_train)
        y_prob = model.predict_proba(X_tune_val)[:, 1]
        return average_precision_score(y_tune_val, y_prob)

    logger.info("Iniciando busca Optuna (%d trials)...", args.trials)
    start = time.time()
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    study.optimize(objective, n_trials=args.trials)
    logger.info("Busca concluida em %.1fs", time.time() - start)
    logger.info("Melhor PR-AUC de validacao: %.4f", study.best_value)
    logger.info("Melhores hiperparametros: %s", study.best_params)

    ensure_dir(config.paths.reports_dir)
    study.trials_dataframe().to_csv(
        config.paths.reports_dir / "lightgbm_tuning_trials.csv", index=False
    )

    # Modelo final: retreina com TODO o treino, usando o pipeline global
    # (ja ajustado no treino completo por scripts/train_models.py).
    logger.info("Retreinando modelo final no conjunto de treino completo...")
    full_pipeline = joblib.load(pipeline_path)
    X_full_train = full_pipeline.transform(full_train_df[feature_cols])
    y_full_train = full_train_df[target_col].to_numpy()

    final_model = lgb.LGBMClassifier(**_lgbm_kwargs(study.best_params, config))
    final_model.fit(X_full_train, y_full_train)

    save_model(final_model, "LightGBM Tuned", config.paths.models_dir)
    logger.info("Concluido: modelo 'LightGBM Tuned' salvo em %s", config.paths.models_dir)


if __name__ == "__main__":
    main()
