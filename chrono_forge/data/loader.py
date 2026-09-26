"""数据载入/导出：CSV、numpy、Python list。

Author: 晨星
"""
from __future__ import annotations

import csv
from typing import Iterable, Optional, Sequence

import numpy as np

from ..core.types import TimeSeries


def from_array(arr: Sequence[float], name: str = "series") -> TimeSeries:
    """从数组/list 构造 TimeSeries。"""
    return TimeSeries(values=np.asarray(arr, dtype=float).ravel(), name=name)


def from_csv(path: str, column: Optional[str] = None, header: bool = True) -> TimeSeries:
    """读取单变量 CSV。column 指定列名；否则取第一列数值。"""
    rows: List[float] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        head = next(reader) if header else None
        idx = 0
        if header and column is not None:
            try:
                idx = head.index(column)  # type: ignore[arg-type]
            except (ValueError, AttributeError):
                idx = 0
        for line in reader:
            if not line:
                continue
            try:
                rows.append(float(line[idx]))
            except (ValueError, IndexError):
                continue
    if not rows:
        raise ValueError(f"CSV 未解析出有效数值: {path}")
    return TimeSeries(values=np.asarray(rows, dtype=float), name=column or "csv")


def to_csv(ts: TimeSeries, path: str) -> None:
    """导出为单列 CSV（含表头 value）。"""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["value"])
        for v in ts.values:
            writer.writerow([f"{v:.6f}"])
