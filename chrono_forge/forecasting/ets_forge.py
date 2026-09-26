"""ETS 封装：statsmodels Holt-Winters 指数平滑（顶级经典 SOTA）。

Author: 晨星
"""
from __future__ import annotations

import numpy as np

from .base import BaseForecaster
from ..core.errors import DependencyError


class ETSForecaster(BaseForecaster):
    """指数平滑（趋势 / 加性季节）。"""

    name = "ets"

    def __init__(self, config=None) -> None:
        super().__init__(config)
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            self._ETS = ExponentialSmoothing
            self.available = True
        except ImportError:
            self.available = False

    def fit(self, y: np.ndarray) -> "ETSForecaster":
        self._ensure_available()
        self.y = np.asarray(y, dtype=float).ravel()
        season = (
            self.config.seasonality
            if (self.config.seasonality > 1 and len(self.y) >= 2 * self.config.seasonality)
            else None
        )
        if season:
            self.model_ = self._ETS(
                self.y, trend="add", seasonal="add", seasonal_periods=season
            ).fit()
        else:
            self.model_ = self._ETS(self.y, trend="add", seasonal=None).fit()
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        return np.asarray(self.model_.forecast(horizon), dtype=float).ravel()
