"""
LLM模型模块
提供嵌入模型和重排序模型的定义
"""

import time
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Union
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BGEEmbeddingModel(BaseEmbedding):
    """
    BGE嵌入模型类，继承自BaseEmbedding
    用于通过API调用BGE-M3嵌入模型获取文本嵌入向量
    """
    api_url: str = Field(description="嵌入模型API URL")
    model_name: str = Field(description="嵌入模型名称")
    api_key: Optional[str] = Field(default=None, description="API密钥")
    headers: Dict[str, str] = Field(description="HTTP请求头")
    timeout: int = Field(default=60, description="请求超时时间")
    max_retries: int = Field(default=3, description="最大重试次数")

    def __init__(self, config: Dict[str, Any]):
        """
        初始化BGE嵌入模型
        
        Args:
            config: 配置字典，包含API URL、模型名称和API密钥
        
        Raises:
            ValueError: 当必要配置参数缺失时
        """
        # 验证必要的配置参数
        required_keys = ["bge_m3_embedding_url", "bge_m3_embedding_name", "llm_api_key"]
        for key in required_keys:
            if key not in config or not config[key]:
                raise ValueError(f"配置缺少必要参数: {key}")

        api_url = config["bge_m3_embedding_url"]
        model_name = config["bge_m3_embedding_name"]
        api_key = config["llm_api_key"]
        timeout = config.get("request_timeout", 60)
        max_retries = config.get("max_retries", 3)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        super().__init__(
            api_url=api_url,
            model_name=model_name,
            api_key=api_key,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries
        )

        logger.info(f"BGEEmbeddingModel 初始化成功，模型: {model_name}")

    def _get_embedding(self, text: str) -> List[float]:
        """
        获取单个文本的嵌入向量
        
        Args:
            text: 输入文本
        
        Returns:
            List[float]: 嵌入向量，如果获取失败则返回空列表
        """
        if not text.strip():
            logger.warning("尝试为空白文本获取嵌入向量")
            return []

        payload = {
            "model": self.model_name,
            "input": text
        }

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )

                response.raise_for_status()
                result = response.json()
                
                if "data" in result and len(result["data"]) > 0:
                    embedding = result["data"][0]["embedding"]
                    if isinstance(embedding, list) and len(embedding) > 0:
                        return embedding
                    else:
                        logger.error(f"嵌入向量格式错误: {embedding}")
                        return []
                else:
                    logger.error(f"API响应格式错误: {result}")
                    return []
                    
            except requests.exceptions.RequestException as e:
                retry_count += 1
                logger.warning(f"嵌入API调用失败，重试 {retry_count}/{self.max_retries}: {e}")
                if retry_count >= self.max_retries:
                    logger.error(f"嵌入API调用最终失败: {e}")
                    return []
                time.sleep(1)  # 重试前等待1秒
            except Exception as e:
                logger.error(f"获取嵌入向量时出现未知错误: {e}")
                return []

        return []

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        获取查询文本的嵌入向量
        
        Args:
            query: 查询文本
        
        Returns:
            List[float]: 嵌入向量
        """
        return self._get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量
        
        Args:
            text: 输入文本
        
        Returns:
            List[float]: 嵌入向量
        """
        return self._get_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的嵌入向量
        
        Args:
            texts: 文本列表
        
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    # 适配新版本 BaseEmbedding 抽象接口（异步）
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询嵌入（委托同步实现）。"""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步获取文本嵌入（委托同步实现）。"""
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步批量获取文本嵌入（委托同步实现）。"""
        return self._get_text_embeddings(texts)


class BGERerankerModel:
    """
    BGE重排序模型类
    用于对检索结果进行重排序
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化BGE重排序模型
        
        Args:
            config: 配置字典
        
        Raises:
            ValueError: 当必要配置参数缺失时
        """
        # 验证必要的配置参数
        required_keys = ["bge_reranker_url", "bge_reranker_name", "llm_api_key"]
        for key in required_keys:
            if key not in config or not config[key]:
                raise ValueError(f"配置缺少必要参数: {key}")

        self.api_url = config["bge_reranker_url"]
        self.model_name = config["bge_reranker_name"]
        self.api_key = config["llm_api_key"]
        self.timeout = config.get("request_timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        self.score_threshold = config.get("reranker_score_threshold", 0.6)

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        logger.info(f"BGERerankerModel 初始化成功，模型: {self.model_name}")

    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
        
        Returns:
            List[Dict[str, Any]]: 重排序结果，包含文档和得分
        """
        if not documents:
            return []

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents
        }

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )

                response.raise_for_status()
                result = response.json()
                
                if "results" in result:
                    # 过滤低分结果
                    filtered_results = []
                    for item in result["results"]:
                        if item.get("score", 0) >= self.score_threshold:
                            filtered_results.append({
                                "document": item.get("document", ""),
                                "score": item.get("score", 0),
                                "index": item.get("index", 0)
                            })
                    
                    # 按得分降序排序
                    filtered_results.sort(key=lambda x: x["score"], reverse=True)
                    if isinstance(top_k, int) and top_k > 0:
                        filtered_results = filtered_results[:top_k]
                    return filtered_results
                else:
                    logger.error(f"重排序API响应格式错误: {result}")
                    return []
                    
            except requests.exceptions.RequestException as e:
                retry_count += 1
                logger.warning(f"重排序API调用失败，重试 {retry_count}/{self.max_retries}: {e}")
                if retry_count >= self.max_retries:
                    logger.error(f"重排序API调用最终失败: {e}")
                    return []
                time.sleep(1)  # 重试前等待1秒
            except Exception as e:
                logger.error(f"重排序时出现未知错误: {e}")
                return []

        return []
