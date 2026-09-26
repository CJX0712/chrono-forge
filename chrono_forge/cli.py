"""ChronoForge CLI 入口。

用法：
  python -m chrono_forge.cli benchmark [--out benchmark.json]
  python -m chrono_forge.cli demo
  python -m chrono_forge.cli forecast --csv data.csv [--horizon 12] [--seasonality 12]

Author: 晨星
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chrono_forge.core.config import load_config
from chrono_forge.core.types import ForecastConfig
from chrono_forge.pipeline import ChronoForgePipeline
from chrono_forge.data.loader import from_csv


def _cmd_benchmark(args) -> int:
    cfg = load_config(ForecastConfig(horizon=args.horizon, seasonality=args.seasonality))
    pipe = ChronoForgePipeline(cfg)
    report = pipe.benchmark()
    print(ChronoForgePipeline.print_table(report))
    print("\n最佳模型（按数据集）：")
    for ds, best in report["best_per_dataset"].items():
        print(f"  {ds:<16} -> {best}")
    print("\n胜场统计：", report["win_counts"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\n已落盘 {args.out}")
    return 0


def _cmd_forecast(args) -> int:
    ts = from_csv(args.csv)
    cfg = load_config(
        ForecastConfig(horizon=args.horizon, seasonality=args.seasonality)
    )
    pipe = ChronoForgePipeline(cfg)
    fc = pipe.forecast(ts.values)
    if not fc:
        print("所有模型均不可用，请检查依赖。")
        return 1
    # 选 sMAPE 最佳（此处无真值，退化为首个可用 / residual_stacking 优先）
    prefer = "residual_stacking" if "residual_stacking" in fc else next(iter(fc))
    print(f"模型数量：{len(fc)}，推荐：{prefer}")
    print("预测（未来 {0} 步）：".format(args.horizon))
    print(", ".join(f"{v:.2f}" for v in fc[prefer]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chrono-forge", description="ChronoForge 时间序列预测系统")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="跨数据集基准对比")
    b.add_argument("--horizon", type=int, default=12)
    b.add_argument("--seasonality", type=int, default=12)
    b.add_argument("--out", type=str, default="benchmark.json")
    b.set_defaults(func=_cmd_benchmark)

    d = sub.add_parser("demo", help="运行 demo（同 benchmark）")
    d.add_argument("--horizon", type=int, default=12)
    d.add_argument("--seasonality", type=int, default=12)
    d.add_argument("--out", type=str, default="benchmark.json")
    d.set_defaults(func=_cmd_benchmark)

    f = sub.add_parser("forecast", help="对 CSV 做预测")
    f.add_argument("--csv", type=str, required=True)
    f.add_argument("--horizon", type=int, default=12)
    f.add_argument("--seasonality", type=int, default=12)
    f.set_defaults(func=_cmd_forecast)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
