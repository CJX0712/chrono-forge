"""错误码体系（E100~E500），统一异常基类。

Author: 晨星
"""
from __future__ import annotations


class ChronoForgeError(Exception):
    """所有 ChronoForge 异常的基类。"""

    code = "E000"

    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(f"[{self.code}] {message}")


class ConfigError(ChronoForgeError):
    """E100 —— 配置非法。"""

    code = "E100"


class DataError(ChronoForgeError):
    """E200 —— 数据校验失败（长度不足、含 NaN 等）。"""

    code = "E200"


class FitError(ChronoForgeError):
    """E300 —— 模型拟合/预测失败。"""

    code = "E300"


class DependencyError(ChronoForgeError):
    """E400 —— 可选顶级依赖缺失，需走离线兜底。"""

    code = "E400"


class PipelineError(ChronoForgeError):
    """E500 —— 流水线编排失败。"""

    code = "E500"
