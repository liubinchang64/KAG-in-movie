import time
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Union
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
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

                if response.status_code == 200:
                    data = response.json()
                    if "data" in data and len(data["data"]) > 0 and "embedding" in data["data"][0]:
                        embedding = data["data"][0]["embedding"]
                        logger.debug(f"成功获取嵌入向量，长度: {len(embedding)}")
                        return embedding
                    else:
                        logger.error(f"嵌入模型响应格式错误: {data}")
                else:
                    logger.error(f"嵌入模型调用失败，状态码: {response.status_code}，响应: {response.text}")
            except requests.exceptions.Timeout:
                logger.error(f"嵌入模型调用超时 (第 {retry_count+1} 次)")
            except requests.exceptions.RequestException as e:
                logger.error(f"嵌入模型调用异常 (第 {retry_count+1} 次): {str(e)}")
            except Exception as e:
                logger.error(f"处理嵌入模型响应时异常: {str(e)}")

            retry_count += 1
            if retry_count < self.max_retries:
                logger.info(f"第 {retry_count} 次重试获取嵌入向量...")
                time.sleep(1)

        logger.error(f"达到最大重试次数 ({self.max_retries})，获取嵌入向量失败")
        return []

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        获取查询文本的嵌入向量
        
        Args:
            query: 查询文本
        
        Returns:
            List[float]: 嵌入向量
        """
        logger.debug(f"获取查询嵌入: {query[:30]}...")
        return self._get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        获取普通文本的嵌入向量
        
        Args:
            text: 输入文本
        
        Returns:
            List[float]: 嵌入向量
        """
        logger.debug(f"获取文本嵌入: {text[:30]}...")
        return self._get_embedding(text)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本嵌入向量
        
        Args:
            texts: 文本列表
        
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        logger.info(f"批量获取 {len(texts)} 个文本的嵌入向量")
        # 这里可以优化为真正的批量API调用，如果API支持的话
        return [self._get_embedding(text) for text in texts]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        异步获取查询文本的嵌入向量
        
        Args:
            query: 查询文本
        
        Returns:
            List[float]: 嵌入向量
        """
        # 注意：这个实现仍然是同步的，要真正异步需要使用aiohttp等库
        logger.debug(f"异步获取查询嵌入: {query[:30]}...")
        return self._get_embedding(query)

    async def _aget_embedding(self, text: str) -> List[float]:
        """
        异步获取单个文本的嵌入向量
        
        Args:
            text: 输入文本
        
        Returns:
            List[float]: 嵌入向量
        """
        # 注意：这个实现仍然是同步的，要真正异步需要使用aiohttp等库
        logger.debug(f"异步获取文本嵌入: {text[:30]}...")
        return self._get_embedding(text)

    async def _aget_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        异步批量获取文本嵌入向量
        
        Args:
            texts: 文本列表
        
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        # 注意：这个实现仍然是同步的，要真正异步需要使用aiohttp等库
        logger.info(f"异步批量获取 {len(texts)} 个文本的嵌入向量")
        return self._get_embeddings(texts)


class BGERerankerModel(BaseModel):
    """
    BGE重排序模型类
    用于通过API调用BGE-Reranker模型对文档进行重排序
    """
    api_url: str = Field(description="重排序模型API URL")
    model_name: str = Field(description="重排序模型名称")
    api_key: Optional[str] = Field(default=None, description="API密钥")
    headers: Dict[str, str] = Field(description="HTTP请求头")
    timeout: int = Field(default=60, description="请求超时时间")
    max_retries: int = Field(default=3, description="最大重试次数")
    score_threshold: float = Field(default=0.0, description="重排序得分阈值，低于此值的结果将被过滤")

    def __init__(self, config: Dict[str, Any]):
        """
        初始化BGE重排序模型
        
        Args:
            config: 配置字典，包含API URL、模型名称和API密钥
        
        Raises:
            ValueError: 当必要配置参数缺失时
        """
        # 验证必要的配置参数
        required_keys = ["bge_reranker_url", "bge_reranker_name", "llm_api_key"]
        for key in required_keys:
            if key not in config or not config[key]:
                raise ValueError(f"配置缺少必要参数: {key}")

        api_url = config["bge_reranker_url"]
        model_name = config["bge_reranker_name"]
        api_key = config["llm_api_key"]
        timeout = config.get("request_timeout", 60)
        max_retries = config.get("max_retries", 3)
        score_threshold = config.get("reranker_score_threshold", 0.0)

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
            max_retries=max_retries,
            score_threshold=score_threshold
        )

        logger.info(f"BGERerankerModel 初始化成功，模型: {model_name}，得分阈值: {score_threshold}")

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        根据查询对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回的top k结果数量
        
        Returns:
            List[Dict[str, Any]]: 重排序后的结果列表，包含索引、文档和得分
        """
        if not query.strip():
            logger.warning("尝试使用空白查询进行重排序")
            return []

        if not documents:
            logger.warning("尝试对空文档列表进行重排序")
            return []

        # 确保top_k是有效的
        top_k = max(1, min(top_k, len(documents)))

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_k": top_k
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

                if response.status_code == 200:
                    data = response.json()
                    if "results" in data:
                        # 验证并处理结果中的score字段
                        processed_results = []
                        for result in data["results"]:
                            if isinstance(result, dict):
                                # 优先检查score字段，其次检查relevance_score字段
                                if "score" in result:
                                    score_field = "score"
                                elif "relevance_score" in result:
                                    score_field = "relevance_score"
                                else:
                                    logger.error(f"重排序结果缺少score或relevance_score字段: {result}")
                                    processed_results.append({
                                        "index": result.get("index", 0),
                                        "document": result.get("document", ""),
                                        "score": 0.0
                                    })
                                    continue

                                try:
                                    # 确保score是数值类型
                                    score = float(result[score_field])
                                    processed_results.append({
                                        "index": result.get("index", 0),
                                        "document": result.get("document", ""),
                                        "score": score
                                    })
                                except ValueError:
                                    logger.error(f"重排序结果中{score_field}类型错误: {result[score_field]}")
                                    processed_results.append({
                                        "index": result.get("index", 0),
                                        "document": result.get("document", ""),
                                        "score": 0.0
                                    })
                            else:
                                logger.error(f"重排序结果格式错误: {result}")
                                processed_results.append({
                                    "index": result.get("index", 0) if isinstance(result, dict) else 0,
                                    "document": result.get("document", "") if isinstance(result, dict) else "",
                                    "score": 0.0
                                })

                        # 按score降序排序结果
                        processed_results.sort(key=lambda x: x["score"], reverse=True)
                        
                        # 应用得分阈值过滤
                        filtered_results = [r for r in processed_results if r["score"] >= self.score_threshold]
                        
                        logger.info(f"成功获取重排序结果，原始结果数: {len(processed_results)}，过滤后结果数: {len(filtered_results)}")
                        return filtered_results[:top_k]
                    else:
                        logger.error(f"重排序模型响应格式错误: {data}")
                else:
                    logger.error(f"重排序模型调用失败，状态码: {response.status_code}，响应: {response.text}")
            except requests.exceptions.Timeout:
                logger.error(f"重排序模型调用超时 (第 {retry_count+1} 次)")
            except requests.exceptions.RequestException as e:
                logger.error(f"重排序模型调用异常 (第 {retry_count+1} 次): {str(e)}")
            except Exception as e:
                logger.error(f"处理重排序模型响应时异常: {str(e)}")

            retry_count += 1
            if retry_count < self.max_retries:
                logger.info(f"第 {retry_count} 次重试重排序...")
                time.sleep(1)

        logger.error(f"达到最大重试次数 ({self.max_retries})，重排序失败")
        # 达到最大重试次数后，返回原始文档，按原顺序
        return [{
            "index": i,
            "document": doc,
            "score": 0.0
        } for i, doc in enumerate(documents[:top_k])]

    def postprocess_nodes(self, nodes: List[TextNode], query: str, top_k: int = 5) -> List[TextNode]:
        """
        对TextNode节点进行重排序后处理
        
        Args:
            nodes: TextNode节点列表
            query: 查询文本
            top_k: 返回的top k结果数量
        
        Returns:
            List[TextNode]: 重排序后的节点列表
        """
        if not nodes:
            logger.warning("尝试对空节点列表进行重排序后处理")
            return []

        # 提取节点文本进行重排序
        node_texts = [node.text for node in nodes]
        reranked_results = self.rerank(query, node_texts, top_k)

        # 根据重排序结果重新组织节点
        reranked_nodes = []
        for result in reranked_results:
            original_index = result["index"]
            if 0 <= original_index < len(nodes):
                # 保留原始节点，但添加重排序得分
                node = nodes[original_index]
                node.metadata["rerank_score"] = result["score"]
                reranked_nodes.append(node)
            else:
                logger.warning(f"重排序结果中的索引 {original_index} 超出节点列表范围")

        logger.info(f"完成节点重排序后处理，返回 {len(reranked_nodes)} 个节点")
        return reranked_nodes