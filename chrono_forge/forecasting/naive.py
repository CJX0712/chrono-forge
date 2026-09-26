"""朴素基线：Naive / SeasonalNaive / Drift —— 纯 numpy，永远可用（离线兜底核心）。

Author: 晨星
"""
from __future__ import annotations

import numpy as np

from .base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """朴素法：未来 = 最后一个观测。"""

    name = "naive"

    def fit(self, y: np.ndarray) -> "NaiveForecaster":
        self.y = np.asarray(y, dtype=float).ravel()
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.y[-1])


class SeasonalNaiveForecaster(BaseForecaster):
    """季节朴素法：重复最近一个完整周期。"""

    name = "seasonal_naive"

    def fit(self, y: np.ndarray) -> "SeasonalNaiveForecaster":
        self.y = np.asarray(y, dtype=float).ravel()
        self._period = self.config.seasonality if self.config.seasonality > 1 else 1
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        block = self.y[-self._period:]
        reps = (horizon + self._period - 1) // self._period
        return np.tile(block, reps)[:horizon]


class DriftForecaster(BaseForecaster):
    """漂移法：按首尾连线的平均斜率外推。"""

    name = "drift"

    def fit(self, y: np.ndarray) -> "DriftForecaster":
        self.y = np.asarray(y, dtype=float).ravel()
        n = len(self.y)
        self._slope = (self.y[-1] - self.y[0]) / max(1, n - 1)
        self._last = self.y[-1]
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        steps = np.arange(1, horizon + 1, dtype=float)
        return self._last + self._slope * steps
