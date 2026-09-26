"""端到端演示：合成数据 -> 跨模型基准 -> 打印表 + 落盘 benchmark.json。

零下载可跑：statsmodels / pmdarima / scikit-learn 任一缺失都会自动降级。

Author: 晨星
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chrono_forge.core.config import load_config
from chrono_forge.core.types import ForecastConfig
from chrono_forge.pipeline import ChronoForgePipeline

# 固定 random_state 保证基线可复现
CONFIG = load_config(ForecastConfig(horizon=12, seasonality=12, cv_folds=3, random_state=42))

PIPE = ChronoForgePipeline(CONFIG)


def main() -> None:
    report = PIPE.benchmark()
    print("=" * 70)
    print("ChronoForge 基准对比（horizon=12, cv_folds=3, random_state=42）")
    print("=" * 70)
    print(ChronoForgePipeline.print_table(report))
    print("\n最佳模型（按数据集）：")
    for ds, best in report["best_per_dataset"].items():
        print(f"  {ds:<16} -> {best}")
    print("\n胜场统计：", report["win_counts"])

    out_path = os.path.join(ROOT, "benchmark.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n已落盘 {out_path}")


if __name__ == "__main__":
    main()
