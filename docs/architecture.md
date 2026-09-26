# ChronoForge 架构设计

**Author**: 晨星 · **版本**: 1.0.0

## 1. 设计目标

一套可运行、可复现、模块化、对标业界 SOTA 的时间序列预测系统，而非原型。核心约束：

- **复用优先**：SOTA 部分一律复用顶级开源，禁止从零自研。
- **离线兜底**：重型/需网络依赖缺失时，纯 numpy/sklearn 降级，保证 demo 零下载可跑。
- **契约先行**：跨模块接口语义统一（「分数越大越异常」类比为「sMAPE 越小越好」），保证公平评测。

## 2. 依赖选型（有据）

| 能力 | 顶级开源 | 版本 | 许可证 | 备注 |
|------|----------|------|--------|------|
| 自动 ARIMA | `pmdarima` | 2.1.1 | MIT | 缺失则 statsmodels 网格兜底 |
| 指数平滑/ETS | `statsmodels` | 0.15.0 | BSD | Holt-Winters |
| 梯度提升元模型 | `scikit-learn` | 1.9.1 | BSD | HistGradientBoosting |
| 数值/表格 | `numpy`/`scipy`/`pandas` | — | BSD | 核心 + 离线兜底 |

## 3. 调用关系（单向无环）

```
cli.py
  └─> pipeline.ChronoForgePipeline
        ├─> data (synthetic/loader)
        ├─> hpo (select_best / tune_ml_lags)
        ├─> training.Trainer
        │     └─> forecasting.* (Naive / AutoARIMA / ETS / ML / ResidualStacking)
        │           └─> preprocess (rolling_splits)
        └─> eval (metrics / backtest)
              └─> preprocess
所有叶子最终依赖 core (types/errors/config/interfaces)
```

## 4. 核心接口契约

- `Forecaster` Protocol：`fit(y) -> self`、`forecast(horizon) -> ndarray`、`available: bool`。
- `ForecastConfig`：全局配置，`ENV_CHRONO_*` 可覆盖。
- 评测主指标 `sMAPE`（对称、尺度无关），跨模型公平对比。

## 5. 创新点：Residual-Stacking 残差堆叠集成

非平稳序列上，单一基模型常存在系统性偏置。ChronoForge 在滚动回测中收集
`(level, trend, season_diff, base_forecast) -> residual` 样本，训练元模型（GBRT）
学习残差映射；生产时对每个前瞻步预测残差并加回基预测：

```
final_forecast_k = base_forecast_k + meta_model(level, trend, season_diff, base_forecast_k)
```

训练/推理特征严格一致，元模型不窥探未来，故完全可落地。

## 6. 性能基线（复现方式）

```bash
python examples/run_demo.py   # 固定 random_state=42，生成 benchmark.json
```

基线与「最佳模型胜场」写入 `benchmark.json` 与 README 性能基线表，供跨版本回归对比。

## 7. 已知限制 / 后续优化

- 多步递归预测在强非平稳下误差会累积；可引入 direct multi-horizon 训练。
- 未接入 Prophet/NeuralProphet（需编译/重型）；如需可加为可选 backend。
- 外生变量（协变量）支持待扩展。
