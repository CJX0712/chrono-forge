"""ChronoForgePipeline —— 端到端编排：评估单数据集 + 跨数据集基准对比。

调用方向单向无环：pipeline -> {data, hpo, training, forecasting, eval} -> core

Author: 晨星
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.config import load_config
from ..core.types import BenchmarkRow, ForecastConfig, TimeSeries
from ..data.synthetic import make_benchmark_datasets
from ..hpo.search import select_best
from ..training.trainer import Trainer

DEFAULT_SEASON_MAP = {
    "trend": 1,
    "seasonal": 12,
    "trend_seasonal": 12,
    "level_shift": 12,
}


class ChronoForgePipeline:
    """时序预测流水线：评估 + 基准。"""

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = load_config(config)

    def evaluate_dataset(
        self, ts: TimeSeries, seasonality: int
    ) -> (List[BenchmarkRow], Optional[str]):
        """对单数据集评估全部模型，返回 (rows, 最佳模型名)。"""
        cfg = replace(self.config, seasonality=seasonality)
        results = Trainer(cfg).run(ts.values)
        rows: List[BenchmarkRow] = []
        for name, entry in results.items():
            if not entry["available"] or entry["score"] is None:
                rows.append(
                    BenchmarkRow(ts.name, name, float("nan"), float("nan"), float("nan"), False)
                )
            else:
                s = entry["score"]
                rows.append(
                    BenchmarkRow(ts.name, name, s["smape"], s["mae"], s["rmse"], True)
                )
        best, _ = select_best(results, self.config.metric)
        return rows, best

    def benchmark(
        self, datasets: Optional[Dict[str, TimeSeries]] = None,
        season_map: Optional[Dict[str, int]] = None,
    ) -> Dict:
        """跨数据集基准对比，返回可序列化结果字典。"""
        if datasets is None:
            datasets = make_benchmark_datasets(self.config.random_state)
        season_map = season_map or DEFAULT_SEASON_MAP
        rows: List[BenchmarkRow] = []
        best_per: Dict[str, Optional[str]] = {}
        for name, ts in datasets.items():
            season = season_map.get(name, self.config.seasonality)
            r, best = self.evaluate_dataset(ts, season)
            rows.extend(r)
            best_per[name] = best
        win = Counter(best_per.values())
        return {
            "config": asdict(self.config),
            "rows": [asdict(x) for x in rows],
            "best_per_dataset": best_per,
            "win_counts": {k: v for k, v in win.items() if k},
        }

    def forecast(
        self, y: np.ndarray, seasonality: Optional[int] = None, horizon: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """训练全部模型并返回各自点预测（含最佳选型）。"""
        cfg = self.config
        if seasonality is not None:
            cfg = replace(cfg, seasonality=seasonality)
        if horizon is not None:
            cfg = replace(cfg, horizon=horizon)
        results = Trainer(cfg).run(np.asarray(y, dtype=float).ravel())
        out: Dict[str, np.ndarray] = {}
        for name, entry in results.items():
            if entry["available"] and entry["forecaster"] is not None:
                try:
                    out[name] = np.asarray(
                        entry["forecaster"].forecast(cfg.horizon), dtype=float
                    )
                except Exception:
                    continue
        return out

    @staticmethod
    def print_table(report: Dict) -> str:
        """格式化打印基准表（固定宽度，避免中文/数字错位）。"""
        lines = []
        header = f"{'dataset':<16}{'method':<20}{'smape':>10}{'mae':>12}{'rmse':>12}{'avail':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        for row in report["rows"]:
            sm = row["smape"]
            sm_s = f"{sm:>10.2f}" if sm == sm else f"{'NaN':>10}"
            lines.append(
                f"{row['dataset']:<16}{row['method']:<20}{sm_s}"
                f"{row['mae']:>12.2f}{row['rmse']:>12.2f}{('Y' if row['available'] else 'N'):>8}"
            )
        return "\n".join(lines)
