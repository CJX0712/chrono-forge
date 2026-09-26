"""eval 层单测：指标数值正确性 + 滚动回测。

Author: 晨星
"""
import numpy as np

from chrono_forge.core.types import ForecastConfig
from chrono_forge.data.synthetic import make_series
from chrono_forge.eval.backtest import backtest
from chrono_forge.eval.metrics import mae, rmse, smape
from chrono_forge.forecasting import NaiveForecaster

Y = make_series(n=80, random_state=5).values


def test_mae_rmse_known():
    a = np.array([1.0, 2.0, 3.0])
    f = np.array([1.0, 1.0, 4.0])
    assert abs(mae(a, f) - (0 + 1 + 1) / 3) < 1e-9
    assert abs(rmse(a, f) - ((0 + 1 + 1) / 3) ** 0.5) < 1e-9


def test_smape_symmetric_and_zero():
    a = np.array([2.0, 4.0])
    f = np.array([4.0, 2.0])
    assert abs(smape(a, f) - 100 * (2 * (2 + 2) / (2 + 4) / 2)) < 1e-9 or abs(smape(a, f) - 100 * 2 * 2 / 6) < 1e-9
    assert smape(a, a) == 0.0


def test_backtest_naive():
    res = backtest(NaiveForecaster(ForecastConfig(horizon=6, cv_folds=2)), Y, 6, 2)
    assert res is not None
    assert 0 <= res["smape"] <= 200
