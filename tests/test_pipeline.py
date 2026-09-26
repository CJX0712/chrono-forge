"""pipeline 层单测：基准对比 + 预测输出。

Author: 晨星
"""
import numpy as np

from chrono_forge.core.config import load_config
from chrono_forge.core.types import ForecastConfig
from chrono_forge.pipeline import ChronoForgePipeline


def test_benchmark_shape():
    pipe = ChronoForgePipeline(load_config(ForecastConfig(horizon=12, cv_folds=2, random_state=1)))
    report = pipe.benchmark()
    assert len(report["rows"]) == 4 * 7  # 4 数据集 × 7 模型
    assert set(report["best_per_dataset"].keys()) == {
        "trend",
        "seasonal",
        "trend_seasonal",
        "level_shift",
    }
    assert all(v for v in report["best_per_dataset"].values())


def test_forecast_returns_models():
    pipe = ChronoForgePipeline(load_config(ForecastConfig(horizon=12, cv_folds=2, random_state=1)))
    y = np.sin(np.arange(80)) + np.arange(80) * 0.02
    fc = pipe.forecast(y, seasonality=12, horizon=12)
    assert len(fc) >= 1
    for v in fc.values():
        assert len(v) == 12 and np.all(np.isfinite(v))


def test_print_table_no_crash():
    pipe = ChronoForgePipeline(load_config(ForecastConfig(horizon=8, cv_folds=2, random_state=1)))
    report = pipe.benchmark()
    table = ChronoForgePipeline.print_table(report)
    assert "dataset" in table and "smape" in table
