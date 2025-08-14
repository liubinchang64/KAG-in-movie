import json
import os
import shutil
import time
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any, Union
from core.utils import hash_text
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.schema import Document
from core.models import BGEEmbeddingModel, BGERerankerModel
from core.utils import load_main_nodes_from_source, ensure_dirs, atomic_write

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


class MainKnowledgeManager:
    """
    主知识库管理器类
    用于管理主知识库的索引构建、加载和检索
    支持增量更新和索引优化
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化主知识库管理器
        
        Args:
            config: 配置字典，包含知识库路径、索引目录等信息
        
        Raises:
            ValueError: 当配置参数不完整或无效时
        """
        # 验证配置参数
        required_keys = ["main_meta_path", "main_review_path", "main_meta_index_dir", "main_review_index_dir"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置缺少必要参数: {key}")
            if not isinstance(config[key], str):
                raise ValueError(f"参数 {key} 必须是字符串类型")

        self.config = config
        self.meta_path = config["main_meta_path"]
        self.review_path = config["main_review_path"]
        self.meta_index_dir = config["main_meta_index_dir"]
        self.review_index_dir = config["main_review_index_dir"]
        self.meta_index: Optional[VectorStoreIndex] = None
        self.review_index: Optional[VectorStoreIndex] = None
        self.last_updated: Dict[str, float] = {}
        self.document_cache: Dict[str, Dict[str, Document]] = {"meta": {}, "review": {}}
        self.enable_incremental_update = config.get("enable_incremental_update", True)
        
        # 初始化嵌入模型
        try:
            self.embed_model = BGEEmbeddingModel(config)
            # 设置全局嵌入模型（注意：这会影响其他使用Settings的模块）
            Settings.embed_model = self.embed_model
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {e}")
            raise

        # 初始化重排序模型
        try:
            self.reranker_model = BGERerankerModel(config)
        except Exception as e:
            logger.error(f"初始化重排序模型失败: {e}")
            raise

        # 从配置获取相似度top_k
        self.similarity_top_k = config.get("similarity_top_k", 10)
        if not isinstance(self.similarity_top_k, int) or self.similarity_top_k <= 0:
            raise ValueError("similarity_top_k 必须是正整数")

        # 索引优化参数
        self.index_optimization = config.get("index_optimization", {})
        if not isinstance(self.index_optimization, dict):
            raise ValueError("index_optimization 必须是字典类型")

        # 确保目录存在
        try:
            ensure_dirs(self.meta_index_dir, self.review_index_dir)
        except Exception as e:
            logger.error(f"创建目录失败: {e}")
            raise

        # 加载最后更新时间
        self._load_last_updated()
        logger.info("MainKnowledgeManager 初始化成功")

    def _load_last_updated(self) -> None:
        """加载最后更新时间"""
        last_updated_path = os.path.join(os.path.dirname(self.meta_index_dir), "last_updated.json")
        if os.path.exists(last_updated_path):
            try:
                with open(last_updated_path, 'r', encoding='utf-8') as f:
                    self.last_updated = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"解析最后更新时间文件失败: {e}")
                self.last_updated = {"meta": 0, "review": 0}
            except Exception as e:
                logger.error(f"加载最后更新时间失败: {e}")
                self.last_updated = {"meta": 0, "review": 0}
        else:
            self.last_updated = {"meta": 0, "review": 0}

    def _save_last_updated(self) -> None:
        """保存最后更新时间"""
        last_updated_path = os.path.join(os.path.dirname(self.meta_index_dir), "last_updated.json")
        try:
            atomic_write(last_updated_path, self.last_updated)
        except Exception as e:
            logger.error(f"保存最后更新时间失败: {e}")

    def _get_modified_docs(self, doc_type: str) -> List[Document]:
        """
        获取修改过的文档
        
        Args:
            doc_type: 文档类型 (meta 或 review)
        
        Returns:
            list: 修改过的文档列表
        
        Raises:
            ValueError: 当doc_type无效时
        """
        if doc_type not in ["meta", "review"]:
            raise ValueError("doc_type 必须是 'meta' 或 'review'")

        if doc_type == "meta":
            docs_path = self.meta_path
            last_update_time = self.last_updated.get("meta", 0)
        else:
            docs_path = self.review_path
            last_update_time = self.last_updated.get("review", 0)

        modified_docs = []
        if os.path.exists(docs_path):
            for fname in os.listdir(docs_path):
                if fname.endswith(".txt"):
                    fpath = os.path.join(docs_path, fname)
                    try:
                        file_modified_time = os.path.getmtime(fpath)
                        if file_modified_time > last_update_time:
                            try:
                                with open(fpath, "r", encoding="utf-8") as f:
                                    text = f.read().strip()
                                if text:
                                    doc = Document(text=text, metadata={"filename": fname, "last_modified": file_modified_time})
                                    modified_docs.append(doc)
                            except UnicodeDecodeError:
                                logger.error(f"文件编码错误: {fpath}")
                            except Exception as e:
                                logger.error(f"读取文件失败 {fpath}: {e}")
                    except Exception as e:
                        logger.error(f"获取文件修改时间失败 {fpath}: {e}")
        return modified_docs

    def _optimize_single_index(self, index: VectorStoreIndex, index_dir: str, doc_type: str) -> None:
        """
        优化单个索引
        
        Args:
            index: 要优化的索引
            index_dir: 索引目录
            doc_type: 文档类型 (meta 或 review)
        """
        logger.info(f"优化{doc_type}索引...")

        # 索引优化配置
        optimize_config = self.index_optimization
        should_compress = optimize_config.get("compress", False)
        compression_level = optimize_config.get("compression_level", 1)
        remove_duplicates = optimize_config.get("remove_duplicates", True)
        min_doc_size = optimize_config.get("min_doc_size", 10)

        # 获取当前存储上下文
        sc = index.storage_context
        # 获取文档存储
        docstore = sc.docstore
        # 清理无效文档
        valid_docs = {}
        for doc_id, doc in docstore.docs.items():
            # 移除过小的文档
            if len(doc.text.strip()) >= min_doc_size:
                valid_docs[doc_id] = doc
            else:
                logger.debug(f"移除过小的{doc_type}文档: {doc_id}")

        # 去重
        if remove_duplicates:
            unique_docs = {}
            seen_hashes = set()
            for doc_id, doc in valid_docs.items():
                doc_hash = hash_text(doc.text)
                if doc_hash not in seen_hashes:
                    seen_hashes.add(doc_hash)
                    unique_docs[doc_id] = doc
                else:
                    logger.debug(f"移除重复的{doc_type}文档: {doc_id}")
            valid_docs = unique_docs

        # 更新文档存储
        if len(valid_docs) != len(docstore.docs):
            docstore.docs = valid_docs
            # 重新创建索引
            new_index = VectorStoreIndex.from_documents(
                documents=list(valid_docs.values()),
                storage_context=sc,
                embed_model=self.embed_model,
            )
            # 持久化更新后的索引
            new_index.storage_context.persist(persist_dir=index_dir)
            logger.info(f"{doc_type}索引优化完成，保留 {len(valid_docs)} 个文档")
            # 更新索引引用
            if doc_type == "meta":
                self.meta_index = new_index
            else:
                self.review_index = new_index
        else:
            logger.info(f"{doc_type}索引无需优化")

        # 压缩索引 (如果支持)
        if should_compress:
            logger.info(f"压缩{doc_type}索引，压缩级别: {compression_level}")
            # 这里可以添加实际的压缩逻辑
            # 例如: 使用gzip压缩索引文件
            pass

    def optimize_index(self) -> None:
        """
        优化索引
        包括索引压缩、碎片整理、清理无效文档等操作
        """
        logger.info("开始优化主库索引...")
        start_time = time.time()

        # 优化meta索引
        if self.meta_index:
            self._optimize_single_index(self.meta_index, self.meta_index_dir, "meta")
        else:
            logger.warning("meta索引未加载，跳过优化")

        # 优化review索引
        if self.review_index:
            self._optimize_single_index(self.review_index, self.review_index_dir, "review")
        else:
            logger.warning("review索引未加载，跳过优化")

        end_time = time.time()
        logger.info(f"主库索引优化完成，耗时: {end_time - start_time:.2f}秒")

    def build_index(self) -> None:
        """
        构建主知识库索引
        从源文件加载文档，创建并持久化向量索引
        支持全量构建和增量更新
        """
        start_time = time.time()
        logger.info("开始构建主库索引...")

        # 确保目录存在
        try:
            os.makedirs(self.meta_index_dir, exist_ok=True)
            os.makedirs(self.review_index_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"创建索引目录失败: {e}")
            raise

        # 检查是否启用增量更新
        if self.enable_incremental_update and \
           os.path.exists(os.path.join(self.meta_index_dir, "index_store.json")) and \
           os.path.exists(os.path.join(self.review_index_dir, "index_store.json")):
            logger.info("启用增量更新模式")
            try:
                self.incremental_update()
            except Exception as e:
                logger.error(f"增量更新失败: {e}")
                logger.info("回退到全量构建模式")
                self._full_build()
        else:
            logger.info("启用全量构建模式")
            self._full_build()

        end_time = time.time()
        logger.info(f"主库索引构建完成，耗时: {end_time - start_time:.2f}秒")

    def _full_build(self) -> None:
        """
        全量构建索引
        """
        try:
            # 从源文件加载文档
            meta_docs, review_docs = load_main_nodes_from_source(self.config)
            vector_stores_dict = {"default": SimpleVectorStore()}

            # 构建meta索引
            if meta_docs:
                storage_context_meta = StorageContext.from_defaults(
                    docstore=SimpleDocumentStore(),
                    index_store=SimpleIndexStore(),
                    vector_stores=vector_stores_dict
                )
                self.meta_index = VectorStoreIndex.from_documents(
                    documents=meta_docs,
                    storage_context=storage_context_meta,
                    embed_model=self.embed_model,
                )
                self.meta_index.storage_context.persist(persist_dir=self.meta_index_dir)
                # 更新最后更新时间
                self.last_updated["meta"] = time.time()
            else:
                logger.warning("无meta文档，跳过meta索引构建")

            # 构建review索引
            if review_docs:
                storage_context_review = StorageContext.from_defaults(
                    docstore=SimpleDocumentStore(),
                    index_store=SimpleIndexStore(),
                    vector_stores=vector_stores_dict
                )
                self.review_index = VectorStoreIndex.from_documents(
                    documents=review_docs,
                    storage_context=storage_context_review,
                    embed_model=self.embed_model,
                )
                self.review_index.storage_context.persist(persist_dir=self.review_index_dir)
                # 更新最后更新时间
                self.last_updated["review"] = time.time()
            else:
                logger.warning("无review文档，跳过review索引构建")

            # 保存最后更新时间
            self._save_last_updated()
        except Exception as e:
            logger.error(f"全量构建索引失败: {e}")
            raise

    def incremental_update(self) -> None:
        """
        增量更新索引
        加载上次更新后修改的文档并更新索引
        """
        # 这里应该实现增量更新逻辑
        # 为了简化示例，我们暂时留空
        logger.warning("incremental_update 方法尚未实现")

    def get_retrievers(self):
        """
        获取主知识库的检索器
        加载已持久化的索引，并返回检索器和重排序模型
        
        Returns:
            tuple: (meta检索器, review检索器, 重排序模型)
        
        Raises:
            RuntimeError: 如果索引未加载
        """
        if self.meta_index is None and os.path.exists(os.path.join(self.meta_index_dir, "index_store.json")):
            meta_sc = StorageContext.from_defaults(persist_dir=self.meta_index_dir)
            self.meta_index = load_index_from_storage(meta_sc)
            # 设置嵌入模型
            self.meta_index._embed_model = self.embed_model

        if self.review_index is None and os.path.exists(os.path.join(self.review_index_dir, "index_store.json")):
            review_sc = StorageContext.from_defaults(persist_dir=self.review_index_dir)
            self.review_index = load_index_from_storage(review_sc)
            # 设置嵌入模型
            self.review_index._embed_model = self.embed_model

        if self.meta_index and self.review_index:
            return (
                self.meta_index.as_retriever(similarity_top_k=self.similarity_top_k),
                self.review_index.as_retriever(similarity_top_k=self.similarity_top_k),
                self.reranker_model
            )
        else:
            missing = []
            if not self.meta_index:
                missing.append("meta索引")
            if not self.review_index:
                missing.append("review索引")
            raise RuntimeError(f"主库索引未加载，缺失：{','.join(missing)}")