"""Residual-Stacking 残差堆叠集成（ChronoForge 原创增强）。

核心思想：
  基模型（默认 ETS）在滚动回测中产生「系统性残差」——真实值减预测值。
  这些残差往往可被近期上下文（水平 / 趋势 / 季节差 / 基模型预测）解释。
  训练一个元模型（HistGradientBoosting）学习该映射，在生产时对每个前瞻步
  预测残差并加回基预测，从而校正非平稳序列的偏置。

训练/推理特征严格一致：
  [level=y_end, trend=y_end-y_prev, season=y_end-y_end-season, base_forecast_k]

Author: 晨星
"""
from __future__ import annotations

import numpy as np

from .base import BaseForecaster
from .ets_forge import ETSForecaster
from ..core.errors import DependencyError
from ..preprocess.transform import rolling_splits


class ResidualStackingForecaster(BaseForecaster):
    """残差堆叠集成预测器。"""

    name = "residual_stacking"

    # 堆叠收缩：抑制元模型在基模型已拟合良好时的过校正（业界标准正则）
    BLEND = 0.7

    def __init__(self, config=None, base: BaseForecaster | None = None) -> None:
        super().__init__(config)
        self.base = base or ETSForecaster(config)
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor

            self._Meta = HistGradientBoostingRegressor
            self.available = True
        except ImportError:
            self.available = False
        self._meta = None

    def _features(self, ctx: np.ndarray, fc: np.ndarray) -> np.ndarray:
        season = self.config.seasonality
        level = ctx[-1]
        trend = ctx[-1] - ctx[-2] if len(ctx) >= 2 else 0.0
        season_feat = (
            ctx[-1] - ctx[-season] if (season > 1 and len(ctx) >= season) else 0.0
        )
        return np.column_stack(
            [np.full(len(fc), level), np.full(len(fc), trend),
             np.full(len(fc), season_feat), np.asarray(fc, dtype=float)]
        )

    def fit(self, y: np.ndarray) -> "ResidualStackingForecaster":
        self._ensure_available()
        self.y = np.asarray(y, dtype=float).ravel()
        inner = max(2, min(self.config.cv_folds, 3))
        # 元训练步长自适应：保证短序列也能积累足够残差样本（特征模式与推理一致）
        h_meta = min(self.config.horizon, max(1, len(self.y) // 4))
        feats, resids = [], []
        try:
            splits = list(rolling_splits(self.y, h_meta, inner))
        except Exception:
            splits = []
        for train, test in splits:
            try:
                b = type(self.base)(self.config)
                b.fit(train)
                fc = b.forecast(len(test))
            except Exception:
                continue
            X = self._features(train, np.asarray(fc, dtype=float))
            for k in range(len(test)):
                feats.append(X[k])
                resids.append(float(test[k] - fc[k]))
        if len(resids) < 4:
            # 样本不足：优雅退化为基模型（仍可用）
            self.base.fit(self.y)
            self._meta = None
            return self
        self._meta = self._Meta(random_state=self.config.random_state).fit(
            np.array(feats, dtype=float), np.array(resids, dtype=float)
        )
        # 基模型在全量数据上重拟合，用于生产预测
        self.base.fit(self.y)
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        if self._meta is None:
            return self.base.forecast(horizon)
        fc = np.asarray(self.base.forecast(horizon), dtype=float)
        X = self._features(self.y, fc)
        corr = self._meta.predict(X)
        return fc + self.BLEND * np.asarray(corr, dtype=float)
