"""Carregamento da configuracao unica do projeto (config/config.yaml).

Todas as listas de colunas e hiperparametros usados por mais de um modulo
devem ser lidos daqui, para eliminar a duplicacao que existia entre
main.py / data_preparation.py / feature_transformation.py na versao anterior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class PathsConfig:
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path
    train_file: str
    test_file: str

    @property
    def train_path(self) -> Path:
        return self.raw_dir / self.train_file

    @property
    def test_path(self) -> Path:
        return self.raw_dir / self.test_file


@dataclass
class ColumnsConfig:
    target: str
    id_columns: list[str]
    categorical_low_cardinality: list[str]
    categorical_high_cardinality: list[str]
    numeric_to_scale: list[str]
    numeric_passthrough: list[str]

    @property
    def all_categorical(self) -> list[str]:
        return [*self.categorical_low_cardinality, *self.categorical_high_cardinality]

    @property
    def all_feature_columns(self) -> list[str]:
        """Colunas que efetivamente entram no modelo (antes do encoding)."""
        return [
            *self.categorical_low_cardinality,
            *self.categorical_high_cardinality,
            *self.numeric_to_scale,
            *self.numeric_passthrough,
        ]


@dataclass
class TargetEncodingConfig:
    smoothing: float
    min_samples_leaf: int


@dataclass
class ImbalanceConfig:
    strategy: str
    undersample_ratio: int


@dataclass
class EvaluationConfig:
    cost_false_positive: float


@dataclass
class Config:
    seed: int
    paths: PathsConfig
    columns: ColumnsConfig
    target_encoding: TargetEncodingConfig
    imbalance: ImbalanceConfig
    velocity_windows: dict
    models: dict
    evaluation: EvaluationConfig
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: str | os.PathLike | None = None) -> Config:
    """Le config/config.yaml e retorna um objeto Config tipado."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    paths_raw = raw["paths"]
    paths = PathsConfig(
        raw_dir=_resolve(paths_raw["raw_dir"]),
        processed_dir=_resolve(paths_raw["processed_dir"]),
        models_dir=_resolve(paths_raw["models_dir"]),
        reports_dir=_resolve(paths_raw["reports_dir"]),
        train_file=paths_raw["train_file"],
        test_file=paths_raw["test_file"],
    )

    columns = ColumnsConfig(**raw["columns"])
    target_encoding = TargetEncodingConfig(**raw["target_encoding"])
    imbalance = ImbalanceConfig(**raw["imbalance"])
    evaluation = EvaluationConfig(**raw["evaluation"])

    return Config(
        seed=raw["seed"],
        paths=paths,
        columns=columns,
        target_encoding=target_encoding,
        imbalance=imbalance,
        velocity_windows=raw["velocity_windows"],
        models=raw["models"],
        evaluation=evaluation,
    )
