"""预测器抽象基类：统一 name/available/config 契约。

Author: 晨星
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.errors import DependencyError
from ..core.types import ForecastConfig


class BaseForecaster(ABC):
    """所有预测器的基类。"""

    name: str = "base"
    available: bool = True

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()

    def _ensure_available(self) -> None:
        if not self.available:
            raise DependencyError(f"[{self.name}] 依赖不可用，已走离线兜底")

    @abstractmethod
    def fit(self, y: np.ndarray) -> "BaseForecaster":
        ...

    @abstractmethod
    def forecast(self, horizon: int) -> np.ndarray:
        ...
