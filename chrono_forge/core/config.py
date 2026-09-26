"""配置加载：默认值 + ENV_CHRONO_* 环境变量覆盖。

Author: 晨星
"""
from __future__ import annotations

import os

from .types import ForecastConfig


def _coerce_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


_ENV_MAP = {
    "horizon": ("ENV_CHRONO_HORIZON", int),
    "seasonality": ("ENV_CHRONO_SEASONALITY", int),
    "cv_folds": ("ENV_CHRONO_CV_FOLDS", int),
    "random_state": ("ENV_CHRONO_RANDOM_STATE", int),
    "metric": ("ENV_CHRONO_METRIC", str),
    "log_transform": ("ENV_CHRONO_LOG", _coerce_bool),
}


def load_config(base: ForecastConfig | None = None) -> ForecastConfig:
    """用环境变量覆盖 base（或默认）配置；任何非法值回退为原值。"""
    cfg = base or ForecastConfig()
    for attr, (env_key, caster) in _ENV_MAP.items():
        raw = os.getenv(env_key)
        if raw is None or raw == "":
            continue
        try:
            setattr(cfg, attr, caster(raw))
        except (ValueError, TypeError):
            # 非法环境值：保留默认值，不中断流程
            continue
    return cfg
