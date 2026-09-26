"""core 层单测：类型、配置、错误码。

Author: 晨星
"""
import os

import numpy as np

from chrono_forge.core.config import load_config
from chrono_forge.core.errors import ConfigError, DataError, DependencyError, FitError, PipelineError
from chrono_forge.core.types import ForecastConfig, TimeSeries


def test_timeseries_ravel():
    ts = TimeSeries(values=[[1.0], [2.0], [3.0]])
    assert ts.values.shape == (3,)
    assert len(ts.index) == 3


def test_config_defaults():
    cfg = ForecastConfig()
    assert cfg.horizon == 12 and cfg.seasonality == 12 and cfg.cv_folds == 3


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("ENV_CHRONO_HORIZON", "24")
    monkeypatch.setenv("ENV_CHRONO_SEASONALITY", "7")
    monkeypatch.setenv("ENV_CHRONO_LOG", "true")
    cfg = load_config()
    assert cfg.horizon == 24
    assert cfg.seasonality == 7
    assert cfg.log_transform is True


def test_config_env_invalid_kept():
    os.environ["ENV_CHRONO_HORIZON"] = "notanint"
    cfg = load_config()
    assert cfg.horizon == 12  # 非法值回退默认
    del os.environ["ENV_CHRONO_HORIZON"]


def test_error_codes():
    assert ConfigError().code == "E100"
    assert DataError().code == "E200"
    assert FitError().code == "E300"
    assert DependencyError().code == "E400"
    assert PipelineError().code == "E500"
