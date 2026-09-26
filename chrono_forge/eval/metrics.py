"""预测误差指标：MAE / RMSE / MAPE / sMAPE / MASE。

全部手动实现，避免与 sklearn.metrics 同名别名递归（历史踩坑）。

Author: 晨星
"""
from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float).ravel()
    f = np.asarray(forecast, dtype=float).ravel()
    return float(np.mean(np.abs(a - f)))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float).ravel()
    f = np.asarray(forecast, dtype=float).ravel()
    return float(np.sqrt(np.mean((a - f) ** 2)))


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float).ravel()
    f = np.asarray(forecast, dtype=float).ravel()
    m = a != 0
    if not np.any(m):
        return float("inf")
    return float(np.mean(np.abs((a[m] - f[m]) / a[m])) * 100.0)


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """对称 MAPE，范围 [0,200]，对尺度不敏感，适合跨数据集对比。"""
    a = np.asarray(actual, dtype=float).ravel()
    f = np.asarray(forecast, dtype=float).ravel()
    denom = np.abs(a) + np.abs(f)
    m = denom > 0
    if not np.any(m):
        return 0.0
    return float(np.mean(2.0 * np.abs(a[m] - f[m]) / denom[m]) * 100.0)


def mase(actual: np.ndarray, forecast: np.ndarray, naive_mae: float) -> float:
    """MASE：相对朴素法（1 步滞后）的尺度无关误差。"""
    if naive_mae <= 0:
        return float("inf")
    return mae(actual, forecast) / naive_mae
