"""类型定义：时序容器、预测配置、预测结果。

Author: 晨星
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class TimeSeries:
    """一条 univariate 时间序列。"""

    values: np.ndarray
    index: Optional[np.ndarray] = None
    name: str = "series"

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float).ravel()
        if self.index is None:
            self.index = np.arange(len(self.values), dtype=float)
        else:
            self.index = np.asarray(self.index, dtype=float).ravel()


@dataclass
class ForecastConfig:
    """全局预测配置；支持 ENV_CHRONO_* 环境变量覆盖（见 core.config）。"""

    horizon: int = 12
    seasonality: int = 12  # 周期长度；<=1 表示无季节
    cv_folds: int = 3  # 滚动回测折数
    random_state: int = 42
    metric: str = "smape"  # 选型/回测主指标（越小越好）
    log_transform: bool = False  # 是否对序列做 log1p 稳定方差


@dataclass
class ForecastResult:
    """单个模型的预测结果 + 回测评分。"""

    method: str
    point: np.ndarray
    fitted: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    score: Optional[float] = None  # 主指标回测均值（越小越好）
    available: bool = True

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, dtype=float).ravel()


@dataclass
class BenchmarkRow:
    """基准对比的一行。"""

    dataset: str
    method: str
    smape: float
    mae: float
    rmse: float
    available: bool = True
