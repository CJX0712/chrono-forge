"""轻量确定性模型选择（不依赖 optuna，保证零下载可跑）。

Author: 晨星
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..eval.backtest import backtest
from ..forecasting.ml_forecaster import MLForecaster


def select_best(
    results: Dict[str, dict], metric: str = "smape"
) -> Tuple[Optional[str], float]:
    """从 Trainer 产出的 results 中选主指标最小者。"""
    best_name, best_val = None, float("inf")
    for name, entry in results.items():
        if not entry.get("available") or entry.get("score") is None:
            continue
        val = entry["score"].get(metric, float("inf"))
        if val < best_val:
            best_val, best_name = val, name
    return best_name, best_val


def tune_ml_lags(
    y: np.ndarray, config, lags_grid=(3, 6, 12, 18, 24)
) -> Tuple[int, float]:
    """在滞后阶网格上回测 ML 预测器，返回 (best_lags, best_smape)。"""
    y = np.asarray(y, dtype=float).ravel()
    best_lag, best_score = int(lags_grid[0]), float("inf")
    for lags in lags_grid:
        f = MLForecaster(config, n_lags=lags)
        if not f.available:
            continue
        bt = backtest(f, y, config.horizon, config.cv_folds, "smape")
        if bt is None:
            continue
        if bt["smape"] < best_score:
            best_score, best_lag = bt["smape"], lags
    return best_lag, best_score
