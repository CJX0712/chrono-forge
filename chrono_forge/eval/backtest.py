"""滚动原点回测：对单个预测器在多个切分上求指标均值。

Author: 晨星
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .metrics import mae, rmse, smape
from ..preprocess.transform import rolling_splits


def backtest(
    forecaster, y: np.ndarray, horizon: int, n_splits: int, metric: str = "smape"
) -> Optional[Dict[str, float]]:
    """返回 {smape, mae, rmse} 均值；任一折失败或样本不足返回 None。"""
    y = np.asarray(y, dtype=float).ravel()
    scores = {"smape": [], "mae": [], "rmse": []}
    try:
        splits = list(rolling_splits(y, horizon, n_splits))
    except Exception:
        return None
    if not splits:
        return None
    for train, test in splits:
        f = type(forecaster)(forecaster.config)  # 克隆，保证无状态
        try:
            f.fit(train)
            fc = np.asarray(f.forecast(len(test)), dtype=float).ravel()
        except Exception:
            continue  # 单折失败跳过，而非整体判死
        if len(fc) != len(test):
            continue
        scores["smape"].append(smape(test, fc))
        scores["mae"].append(mae(test, fc))
        scores["rmse"].append(rmse(test, fc))
    if not scores["smape"]:
        return None
    return {k: float(np.mean(v)) for k, v in scores.items()}
