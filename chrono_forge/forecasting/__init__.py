"""forecasting —— 领域模块：朴素基线、auto-ARIMA、ETS、ML、残差堆叠集成。

Author: 晨星
"""
from .base import BaseForecaster
from .naive import NaiveForecaster, SeasonalNaiveForecaster, DriftForecaster
from .arima_forge import AutoARIMAForecaster
from .ets_forge import ETSForecaster
from .ml_forecaster import MLForecaster
from .stacking import ResidualStackingForecaster
from .factory import (
    create_forecasters,
    available_auto_arima,
    available_ets,
    available_ml,
    available_stacking,
)

__all__ = [
    "BaseForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "DriftForecaster",
    "AutoARIMAForecaster",
    "ETSForecaster",
    "MLForecaster",
    "ResidualStackingForecaster",
    "create_forecasters",
    "available_auto_arima",
    "available_ets",
    "available_ml",
    "available_stacking",
]
