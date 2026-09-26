"""ChronoForge —— 时间序列预测系统。

随机创新的顶级 AI 系统：复用 statsmodels / pmdarima / scikit-learn 三大顶级开源，
并以 Residual-Stacking（残差堆叠集成）作为原创增强，专治非平稳序列的系统性偏置。

Author: 晨星
"""
from .core.types import TimeSeries, ForecastConfig, ForecastResult
from .core.errors import (
    ChronoForgeError,
    ConfigError,
    DataError,
    FitError,
    DependencyError,
    PipelineError,
)
from .core.config import load_config
from .core.interfaces import Forecaster

__version__ = "1.0.0"
__author__ = "晨星"

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
    "__version__",
    "__author__",
]
