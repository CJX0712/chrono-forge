"""接口协议：Forecaster 契约（输入/输出语义统一）。

约定：
  - fit(y) 接收 1D float 数组，返回 self。
  - forecast(horizon) 返回长度 horizon 的点预测（顺序与输入同方向）。
  - available 标记顶级依赖是否就绪；不可用时应抛出 DependencyError。
  - scores 跨模型可公平比较：均按「越小越好」的回测主指标。

Author: 晨星
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Forecaster(Protocol):
    """预测器协议。"""

    name: str
    available: bool

    def fit(self, y: np.ndarray) -> "Forecaster":
        ...

    def forecast(self, horizon: int) -> np.ndarray:
        ...
