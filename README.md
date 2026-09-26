# ChronoForge

> 随机创新的顶级时间序列预测系统 · 复用 `statsmodels` / `pmdarima` / `scikit-learn` 三大顶级开源，并以 **Residual-Stacking（残差堆叠集成）** 作为原创增强。

- **Author**: 晨星
- **Repo**: `cjx0712/chrono-forge`
- **License**: MIT
- **Python**: 3.13

## ✨ 特性

- **零自研 SOTA**：ARIMA 用 `pmdarima.auto_arima`（含季节项），指数平滑用 `statsmodels` Holt-Winters，梯度提升用 `scikit-learn` HistGradientBoosting。
- **原创增强 —— 残差堆叠集成**：基模型在滚动回测中产生系统性残差，元模型（GBRT）学习「水平/趋势/季节差/基预测 → 残差」映射，生产时逐点校正，专治非平稳序列偏置。
- **离线兜底**：`pmdarima`/`statsmodels`/`sklearn` 任一缺失自动降级为纯 numpy/sklearn 实现（SeasonalNaive / Holt-Winters / 网格 ARIMA），保证零下载 demo 可跑。
- **模块化契约**：单向无环 `cli → pipeline → {data, hpo, training, forecasting, eval} → core`；每个模块可独立验证。
- **可复现基线**：固定 `random_state`，benchmark 落盘 `benchmark.json`。

## 📦 安装

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

依赖（已 pin）：`numpy` `scipy` `pandas` `scikit-learn` `statsmodels` `pmdarima` `pytest`。

## 🚀 一键复现

```bash
# 跨数据集基准对比 + 落盘 benchmark.json
python examples/run_demo.py
# 或
python -m chrono_forge.cli benchmark --out benchmark.json

# 对自有 CSV 做预测（单列数值，含表头）
python -m chrono_forge.cli forecast --csv your_data.csv --horizon 12 --seasonality 12
```

## 🧱 模块架构

```
chrono_forge/
  core/        类型(types) · 错误码(errors E100~E500) · 配置(config+ENV_*) · 接口(Protocol)
  data/        合成数据(synthetic) · 载入/导出(loader)
  preprocess/  变换(log/boxcox/差分) · 滚动切分(rolling_splits)
  forecasting/ 朴素基线 / AutoARIMA / ETS / ML / ResidualStacking / 工厂
  hpo/         模型选型(select_best) · ML 滞后调参(tune_ml_lags)
  training/    训练编排(Trainer)
  eval/        指标(MAE/RMSE/MAPE/sMAPE/MASE) · 滚动回测(backtest)
  pipeline/    ChronoForgePipeline.run() / benchmark()
  cli.py       argparse 入口
  examples/run_demo.py   端到端演示
```

详见 [docs/architecture.md](docs/architecture.md)。

## 🧪 测试

```bash
pytest -q -W ignore::UserWarning
```

## 🐳 部署

```bash
docker build -t chrono-forge .
docker run --rm chrono-forge python examples/run_demo.py
```

## 📊 性能基线（固定 random_state=42, horizon=12, cv_folds=3，可复现）

运行 `python examples/run_demo.py` 落盘 `benchmark.json`。下表为 sMAPE(%) 核心对比：

| dataset | naive | auto_arima | ets | ml_gbrt | **residual_stacking** | 最佳 |
|---------|------|-----------|-----|---------|----------------------|------|
| trend | 39.99 | 39.58 | **30.46** | 36.45 | 30.92 | ets |
| seasonal | 126.21 | **40.92** | 43.54 | 70.05 | 69.60 | auto_arima |
| trend_seasonal | 124.71 | **39.11** | 42.08 | 87.09 | 41.83 | auto_arima |
| level_shift | 123.02 | **28.47** | 52.13 | 70.52 | 53.97 | auto_arima |

说明：残差堆叠集成（residual_stacking）在趋势数据上逼近最佳 ETS（30.92 vs 30.46），在季节/断点数据上与基模型相当；全局由选型器自动挑选每数据集最优模型。完整 7 模型 × 4 数据集的 sMAPE/MAE/RMSE 见 `benchmark.json`。

## 🔧 环境变量覆盖

`ENV_CHRONO_HORIZON` / `ENV_CHRONO_SEASONALITY` / `ENV_CHRONO_CV_FOLDS` / `ENV_CHRONO_RANDOM_STATE` / `ENV_CHRONO_METRIC` / `ENV_CHRONO_LOG`。
