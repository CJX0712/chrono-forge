"""工厂：构造全套预测器 + 顶级依赖可用性探测（离线兜底开关）。

Author: 晨星
"""
from __future__ import annotations

from typing import Callable, List

from .base import BaseForecaster
from .naive import NaiveForecaster, SeasonalNaiveForecaster, DriftForecaster
from .arima_forge import AutoARIMAForecaster
from .ets_forge import ETSForecaster
from .ml_forecaster import MLForecaster
from .stacking import ResidualStackingForecaster
from ..core.types import ForecastConfig


def create_forecasters(config: ForecastConfig | None = None) -> List[BaseForecaster]:
    """返回全部预测器实例（含离线兜底基类）。"""
    cfg = config or ForecastConfig()
    return [
        NaiveForecaster(cfg),
        SeasonalNaiveForecaster(cfg),
        DriftForecaster(cfg),
        AutoARIMAForecaster(cfg),
        ETSForecaster(cfg),
        MLForecaster(cfg),
        ResidualStackingForecaster(cfg),
    ]


def _probe(import_path: str) -> bool:
    import importlib

    try:
        importlib.import_module(import_path)
        return True
    except ImportError:
        return False


def available_auto_arima() -> bool:
    return _probe("pmdarima") or _probe("statsmodels")


def available_ets() -> bool:
    return _probe("statsmodels")


def available_ml() -> bool:
    return _probe("sklearn")


def available_stacking() -> bool:
    return available_ets() and available_ml()
