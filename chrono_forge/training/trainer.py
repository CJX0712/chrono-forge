"""训练编排：拟合全部预测器，产出 name -> {available, score, forecaster}。

Author: 晨星
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.types import ForecastConfig
from ..eval.backtest import backtest
from ..forecasting.factory import create_forecasters


class Trainer:
    """在一组数据上训练并评测全部候选模型。"""

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()

    def run(self, y: np.ndarray) -> Dict[str, dict]:
        y = np.asarray(y, dtype=float).ravel()
        out: Dict[str, dict] = {}
        for f in create_forecasters(self.config):
            entry = {"available": f.available, "score": None, "forecaster": None}
            if not f.available:
                out[f.name] = entry
                continue
            bt = backtest(f, y, self.config.horizon, self.config.cv_folds, self.config.metric)
            if bt is None:
                entry["available"] = False
            else:
                try:
                    f.fit(y)
                    entry["forecaster"] = f
                    entry["score"] = bt
                except Exception:
                    entry["available"] = False
            out[f.name] = entry
        return out
