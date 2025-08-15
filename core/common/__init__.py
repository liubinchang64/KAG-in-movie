"""
公共工具模块
对外导出配置、文本与文件工具
"""

from .config import ConfigManager, get_config_manager, load_config
# 如需暴露更多工具，可在此补充

__all__ = [
	"ConfigManager",
	"get_config_manager",
	"load_config",
]
