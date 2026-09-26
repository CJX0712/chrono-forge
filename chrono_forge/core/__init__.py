"""core —— 契约层：类型、错误码、配置、接口协议。

Author: 晨星
"""
from .types import TimeSeries, ForecastConfig, ForecastResult
from .errors import (
    ChronoForgeError,
    ConfigError,
    DataError,
    FitError,
    DependencyError,
    PipelineError,
)
from .config import load_config
from .interfaces import Forecaster

__all__ = [
    "TimeSeries",
    "ForecastConfig",
    "ForecastResult",
    "ChronoForgeError",
    "ConfigError",
    "DataError",
    "FitError",
    "DependencyError",
    "PipelineError",
    "load_config",
    "Forecaster",
]
