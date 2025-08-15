"""
主知识库包装模块
与旧路径 `core.main_kb` 兼容
"""

# 保持向后兼容：直接复用原实现
from core.main_kb import MainKnowledgeManager

__all__ = ["MainKnowledgeManager"]
