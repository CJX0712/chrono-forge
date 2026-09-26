"""变换与切分工具：方差稳定、差分、滚动原点回测切分。

Author: 晨星
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np

from ..core.errors import DataError


def log_transform(y: np.ndarray) -> np.ndarray:
    """log1p：要求 y > -1。"""
    y = np.asarray(y, dtype=float)
    if np.any(y <= -1.0):
        raise DataError("log_transform 要求所有值 > -1")
    return np.log1p(y)


def exp_transform(y: np.ndarray) -> np.ndarray:
    """log1p 的逆。"""
    return np.expm1(np.asarray(y, dtype=float))


def boxcox_transform(y: np.ndarray, lmbda: float | None = None) -> Tuple[np.ndarray, float]:
    """Box-Cox（依赖 scipy）；lmbda=None 时自动估计。"""
    try:
        from scipy.stats import boxcox
    except ImportError as exc:  # pragma: no cover
        raise DataError("Box-Cox 需要 scipy") from exc
    y = np.asarray(y, dtype=float)
    if np.any(y <= 0.0):
        raise DataError("Box-Cox 要求所有值 > 0")
    out, fitted = boxcox(y, lmbda=lmbda)
    return out, float(fitted)


def inv_boxcox_transform(y: np.ndarray, lmbda: float) -> np.ndarray:
    """Box-Cox 逆变换。"""
    y = np.asarray(y, dtype=float)
    if abs(lmbda) < 1e-8:
        return np.exp(y)
    return np.power(y * lmbda + 1.0, 1.0 / lmbda)


def difference(y: np.ndarray, lag: int = 1) -> np.ndarray:
    """一阶差分。"""
    y = np.asarray(y, dtype=float)
    return y[lag:] - y[:-lag]


def inv_difference(diff: np.ndarray, last: float, lag: int = 1) -> np.ndarray:
    """差分逆运算（递归还原）。"""
    diff = np.asarray(diff, dtype=float)
    out = np.empty(len(diff) + lag, dtype=float)
    out[:lag] = last
    for i in range(len(diff)):
        out[lag + i] = out[lag + i - lag] + diff[i]
    return out[lag:]


def rolling_splits(
    y: np.ndarray, horizon: int, n_splits: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """滚动原点切分：产出 (train, test) 对，test 长度 = horizon。"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    min_train = max(2 * horizon, horizon + 2)
    if n < min_train + horizon:
        raise DataError(
            f"序列长度 {n} 不足以做 {n_splits} 折 horizon={horizon} 回测"
        )
    step = (n - min_train - horizon) // max(1, n_splits)
    step = max(1, step)
    end = min_train
    produced = 0
    while end + horizon <= n and produced < n_splits:
        train = y[:end]
        test = y[end : end + horizon]
        yield train, test
        end += step
        produced += 1
