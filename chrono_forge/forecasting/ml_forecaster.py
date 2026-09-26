"""ML 预测器：以滞后特征训练 HistGradientBoosting（scikit-learn 顶级集成）递归多步预测。

Author: 晨星
"""
from __future__ import annotations

import numpy as np

from .base import BaseForecaster
from ..core.errors import DependencyError


class MLForecaster(BaseForecaster):
    """基于滞后特征的梯度提升时序预测器。"""

    name = "ml_gbrt"

    def __init__(self, config=None, n_lags: int = 12) -> None:
        super().__init__(config)
        self.n_lags = n_lags
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor

            self._Est = HistGradientBoostingRegressor
            self.available = True
        except ImportError:
            self.available = False

    def fit(self, y: np.ndarray) -> "MLForecaster":
        self._ensure_available()
        self.y = np.asarray(y, dtype=float).ravel()
        n_lags = min(self.n_lags, len(self.y) // 3, 20)
        self._n_lags = max(1, n_lags)
        X, target = self._matrix(self.y)
        if len(target) < 1:
            raise DependencyError("[ml_gbrt] 样本不足以构造滞后矩阵")
        self.model_ = self._Est(random_state=self.config.random_state).fit(X, target)
        return self

    def _matrix(self, y: np.ndarray):
        nl = self._n_lags
        X, target = [], []
        for i in range(nl, len(y)):
            X.append(y[i - nl : i])
            target.append(y[i])
        return np.array(X, dtype=float), np.array(target, dtype=float)

    def forecast(self, horizon: int) -> np.ndarray:
        buf = list(self.y[-self._n_lags:])
        out = []
        for _ in range(horizon):
            x = np.array(buf[-self._n_lags :], dtype=float).reshape(1, -1)
            nxt = float(self.model_.predict(x)[0])
            out.append(nxt)
            buf.append(nxt)
        return np.array(out, dtype=float)
