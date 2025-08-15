"""
知识库基础接口
定义主/临时知识库应实现的通用能力
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseKnowledgeManager(ABC):
	"""知识库管理器基础接口"""
	def __init__(self, config: Dict[str, Any]):
		self.config = config

	@abstractmethod
	def build_index(self) -> None:
		"""构建索引"""
		...

	@abstractmethod
	def get_retrievers(self) -> Tuple[Any, Any, Any]:
		"""返回(meta_retriever, review_retriever, reranker)"""
		...

	@abstractmethod
	def clear(self) -> None:
		"""清理索引与缓存"""
		...
