"""合成时序生成：趋势 + 季节 + 噪声 + 可选结构性断点。

固定 random_state 保证可复现；用于零下载 demo 与基准对比。

Author: 晨星
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..core.types import TimeSeries


def make_series(
    n: int = 240,
    trend: float = 0.05,
    season_amp: float = 10.0,
    season_period: int = 12,
    noise: float = 2.0,
    level_shift: bool = False,
    random_state: int = 42,
    name: str = "synthetic",
) -> TimeSeries:
    """生成一条含趋势与季节的合成序列。"""
    rng = np.random.default_rng(random_state)
    t = np.arange(n, dtype=float)
    y = trend * t
    y = y + season_amp * np.sin(2.0 * np.pi * t / max(1, season_period))
    if level_shift:
        k = n // 2
        y[k:] = y[k:] + 30.0  # 结构性断点，考验非平稳建模
    y = y + noise * rng.standard_normal(n)
    return TimeSeries(values=y, name=name)


def make_benchmark_datasets(random_state: int = 42) -> Dict[str, TimeSeries]:
    """构造多 regime 基准集，覆盖平稳/趋势/季节/断点四类典型场景。"""
    rng = np.random.default_rng(random_state)
    base_seed = int(rng.integers(1, 9999))
    datasets: Dict[str, TimeSeries] = {
        "trend": make_series(
            n=240, trend=0.08, season_amp=0.0, noise=1.5,
            random_state=base_seed, name="trend",
        ),
        "seasonal": make_series(
            n=240, trend=0.0, season_amp=12.0, season_period=12, noise=1.5,
            random_state=base_seed + 1, name="seasonal",
        ),
        "trend_seasonal": make_series(
            n=240, trend=0.05, season_amp=10.0, season_period=12, noise=2.0,
            random_state=base_seed + 2, name="trend_seasonal",
        ),
        "level_shift": make_series(
            n=240, trend=0.03, season_amp=8.0, season_period=12, noise=2.0,
            level_shift=True, random_state=base_seed + 3, name="level_shift",
        ),
    }
    return datasets
