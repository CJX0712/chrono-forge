"""forecasting 层单测：朴素基线 + 顶级库封装 + 残差堆叠。

Author: 晨星
"""
import numpy as np

from chrono_forge.core.types import ForecastConfig
from chrono_forge.data.synthetic import make_series
from chrono_forge.forecasting import (
    AutoARIMAForecaster,
    DriftForecaster,
    ETSForecaster,
    MLForecaster,
    NaiveForecaster,
    ResidualStackingForecaster,
    SeasonalNaiveForecaster,
    available_auto_arima,
    available_ets,
    available_ml,
    available_stacking,
)

CFG = ForecastConfig(horizon=12, seasonality=12, cv_folds=3, random_state=42)
Y = make_series(n=120, random_state=3).values


def test_naive_family():
    for cls in (NaiveForecaster, SeasonalNaiveForecaster, DriftForecaster):
        f = cls(CFG)
        f.fit(Y)
        fc = f.forecast(12)
        assert len(fc) == 12 and np.all(np.isfinite(fc))


def test_auto_arima_runs():
    f = AutoARIMAForecaster(CFG)
    if not f.available:
        assert not available_auto_arima()
        return
    f.fit(Y)
    fc = f.forecast(12)
    assert len(fc) == 12 and np.all(np.isfinite(fc))


def test_ets_runs():
    f = ETSForecaster(CFG)
    assert f.available == available_ets()
    f.fit(Y)
    fc = f.forecast(12)
    assert len(fc) == 12 and np.all(np.isfinite(fc))


def test_ml_runs():
    f = MLForecaster(CFG, n_lags=12)
    assert f.available == available_ml()
    f.fit(Y)
    fc = f.forecast(12)
    assert len(fc) == 12 and np.all(np.isfinite(fc))


def test_stacking_runs_and_meta_trained():
    f = ResidualStackingForecaster(CFG)
    assert f.available == (available_ets() and available_ml())
    f.fit(Y)
    fc = f.forecast(12)
    assert len(fc) == 12 and np.all(np.isfinite(fc))
    # 基线足够时元模型应被训练
    if available_ets() and available_ml():
        assert f._meta is not None


def test_availability_probes_consistent():
    assert available_auto_arima() == (AutoARIMAForecaster(CFG).available)
    assert available_stacking() == (ResidualStackingForecaster(CFG).available)
