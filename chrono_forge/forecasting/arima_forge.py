"""Auto-ARIMA 封装：优先 pmdarima（顶级 auto-ARIMA），缺失则 statsmodels 网格兜底。

Author: 晨星
"""
from __future__ import annotations

import numpy as np

from .base import BaseForecaster
from ..core.errors import DependencyError


class AutoARIMAForecaster(BaseForecaster):
    """自动 ARIMA（含可选季节项）。"""

    name = "auto_arima"

    def __init__(self, config=None) -> None:
        super().__init__(config)
        try:
            import pmdarima  # type: ignore

            self._pmdarima = pmdarima
            self._backend = "pmdarima"
            self.available = True
        except ImportError:
            try:
                import statsmodels  # type: ignore

                self._sm = statsmodels
                self._backend = "statsmodels"
                self.available = True
            except ImportError:
                self.available = False
                self._backend = "none"

    def fit(self, y: np.ndarray) -> "AutoARIMAForecaster":
        self._ensure_available()
        self.y = np.asarray(y, dtype=float).ravel()
        if self._backend == "pmdarima":
            seasonal = self.config.seasonality > 1
            self.model_ = self._pmdarima.auto_arima(
                self.y,
                seasonal=seasonal,
                m=self.config.seasonality,
                suppress_warnings=True,
                error_action="ignore",
                random_state=self.config.random_state,
                stepwise=True,
                max_order=6,
                max_p=3,
                max_q=3,
                max_P=1,
                max_Q=1,
                max_D=1,
                n_fits=10,
            )
        else:
            self.model_ = self._fit_sm_arima(self.y)
        return self

    def _fit_sm_arima(self, y: np.ndarray):
        from statsmodels.tsa.arima.model import ARIMA

        best, best_aic = None, float("inf")
        for p in range(0, 4):
            for q in range(0, 4):
                try:
                    m = ARIMA(y, order=(p, 1, q)).fit()
                    if m.aic < best_aic:
                        best_aic, best = m.aic, m
                except Exception:
                    continue
        if best is None:
            raise DependencyError("[auto_arima] statsmodels 网格拟合全部失败")
        return best

    def forecast(self, horizon: int) -> np.ndarray:
        if self._backend == "pmdarima":
            out = self.model_.predict(horizon)
        else:
            out = self.model_.forecast(horizon)
        return np.asarray(out, dtype=float).ravel()
