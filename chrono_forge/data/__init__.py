"""data —— 合成数据生成 + 文件/数组载入。

Author: 晨星
"""
from .synthetic import make_series, make_benchmark_datasets
from .loader import from_csv, from_array, to_csv

__all__ = ["make_series", "make_benchmark_datasets", "from_csv", "from_array", "to_csv"]
