"""data 层单测：合成数据可复现 + 载入导出。

Author: 晨星
"""
import numpy as np

from chrono_forge.data.loader import from_array, from_csv, to_csv
from chrono_forge.data.synthetic import make_benchmark_datasets, make_series


def test_synthetic_deterministic():
    a = make_series(n=100, random_state=7)
    b = make_series(n=100, random_state=7)
    assert np.allclose(a.values, b.values)


def test_benchmark_datasets_count():
    ds = make_benchmark_datasets()
    assert set(ds.keys()) == {"trend", "seasonal", "trend_seasonal", "level_shift"}
    assert all(len(ts.values) == 240 for ts in ds.values())


def test_loader_roundtrip(tmp_path):
    y = np.array([1.0, 2.0, 3.0, 4.0])
    ts = from_array(y, name="t")
    path = tmp_path / "out.csv"
    to_csv(ts, str(path))
    back = from_csv(str(path))
    assert np.allclose(back.values, y)
