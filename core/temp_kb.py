import os
import json
import shutil
import logging
import time
import os
from typing import List, Dict, Any, Optional, Tuple, Set
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.core.schema import TextNode
from core.utils import auto_classify_and_cache, ensure_dirs, load_nodes_from_cache
from core.models import BGEEmbeddingModel, BGERerankerModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# 获取logger
logger = logging.getLogger(__name__)


class TempKnowledgeManager:
    """
    临时知识库管理器类
    用于管理临时知识库的索引构建、加载和检索
    支持增量更新和索引优化
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化临时知识库管理器
        
        Args:
            config: 配置字典，包含知识库路径、索引目录等信息
        """
        # 验证配置参数
        required_config = ["temp_meta_path", "temp_review_path", "temp_meta_index_dir", "temp_review_index_dir"]
        for key in required_config:
            if key not in config:
                raise ValueError(f"配置缺少必要参数: {key}")

        self.config = config
        self.meta_nodes_path = config["temp_meta_path"]
        self.review_nodes_path = config["temp_review_path"]
        self.meta_index_dir = config["temp_meta_index_dir"]
        self.review_index_dir = config["temp_review_index_dir"]
        self.meta_index: Optional[VectorStoreIndex] = None
        self.review_index: Optional[VectorStoreIndex] = None
        self.meta_nodes: List[TextNode] = []
        self.review_nodes: List[TextNode] = []
        self.last_updated: Dict[str, float] = {}
        self.document_cache: Dict[str, Dict[str, Any]] = {"meta": {}, "review": {}}
        self.enable_incremental_update = config.get("enable_incremental_update", True)
        # 确保缓存目录存在
        ensure_dirs(os.path.dirname(self.meta_nodes_path), os.path.dirname(self.review_nodes_path))
        # 初始化嵌入模型
        self.embed_model = BGEEmbeddingModel(config)
        # 设置全局嵌入模型
        Settings.embed_model = self.embed_model
        # 初始化重排序模型
        self.reranker_model = BGERerankerModel(config)
        # 从配置获取相似度top_k
        self.similarity_top_k = self.config.get("similarity_top_k", 10)
        # 索引优化参数
        self.index_optimization = config.get("index_optimization", {})
        # 加载最后更新时间
        self._load_last_updated()
        logger.info("临时知识库管理器初始化成功")

    def _load_last_updated(self):
        """
        加载最后更新时间
        """
        last_updated_path = os.path.join(os.path.dirname(self.meta_index_dir), "temp_last_updated.json")
        if os.path.exists(last_updated_path):
            try:
                with open(last_updated_path, 'r', encoding='utf-8') as f:
                    self.last_updated = json.load(f)
            except Exception as e:
                logger.error(f"加载最后更新时间失败: {e}")
        else:
            self.last_updated = {"meta": 0, "review": 0}

    def _save_last_updated(self):
        """
        保存最后更新时间
        """
        last_updated_path = os.path.join(os.path.dirname(self.meta_index_dir), "temp_last_updated.json")
        try:
            with open(last_updated_path, 'w', encoding='utf-8') as f:
                json.dump(self.last_updated, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存最后更新时间失败: {e}")

    def upload_files(self, paths):
        """
        上传文件或文件夹到临时知识库
        支持多文件和文件夹路径输入
        
        Args:
            paths: 文件或文件夹路径列表
        """
        if not paths:
            logger.warning("未提供上传路径")
            return
    
        # 确保缓存目录存在
        ensure_dirs(os.path.dirname(self.meta_nodes_path), os.path.dirname(self.review_nodes_path))
    
        # 处理每个上传路径
        for path in paths:
            if os.path.exists(path):
                file_size = os.path.getsize(path) / 1024  # KB
                file_type = os.path.splitext(path)[1]
                logger.info(f"开始上传文件：{path}，类型：{file_type}，大小：{file_size:.2f}KB")
                
                if os.path.isdir(path):
                    # 处理目录
                    auto_classify_and_cache(path, self.meta_nodes_path, self.review_nodes_path)
                else:
                    # 处理单文件
                    from llama_index.core.schema import TextNode
                    import hashlib
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                        node = TextNode(text=text, id_=hashlib.md5(text.encode()).hexdigest())
                        data = [node.to_dict()]
                        
                        # 按文件名关键词归类
                        if any(k in os.path.basename(path).lower() for k in ("meta", "metadata", "info", "信息")):
                            save_path = self.meta_nodes_path
                            node_type = "meta"
                        else:
                            save_path = self.review_nodes_path
                            node_type = "review"
                        
                        # 保存节点
                        with open(save_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        logger.info(f"单文件上传完成，保存为{node_type}节点到{save_path}")
                    except Exception as e:
                        logger.error(f"处理单文件{path}失败: {e}")
                        continue
                
                # 重新加载节点
                self.meta_nodes = load_nodes_from_cache(self.meta_nodes_path)
                self.review_nodes = load_nodes_from_cache(self.review_nodes_path)
                logger.info(f"文件上传完成：{path}，生成meta节点数：{len(self.meta_nodes)}，review节点数：{len(self.review_nodes)}")
                # 更新最后更新时间
                current_time = time.time()
                self.last_updated["meta"] = current_time
                self.last_updated["review"] = current_time
                self._save_last_updated()
            else:
                logger.warning(f"上传路径不存在: {path}")
    
        logger.info("所有上传路径处理完成")

    def _create_storage_context(self) -> StorageContext:
        """
        创建存储上下文
        
        Returns:
            StorageContext: 存储上下文对象
        """
        return StorageContext.from_defaults(
            docstore=SimpleDocumentStore(),
            index_store=SimpleIndexStore(),
            vector_store=SimpleVectorStore()
        )

    def clear(self):
        """
        清除临时知识库
        包括缓存节点和索引目录
        """
        import shutil
        # 清理缓存节点
        for p in [self.meta_nodes_path, self.review_nodes_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                logger.error(f"清理缓存文件{p}失败: {e}")
        # 清理索引目录
        for d in [self.meta_index_dir, self.review_index_dir]:
            try:
                if os.path.exists(d):
                    shutil.rmtree(d, ignore_errors=True)
            except Exception as e:
                logger.error(f"清理索引目录{d}失败: {e}")
        self.meta_index = None
        self.review_index = None
        self.meta_nodes = []
        self.review_nodes = []
        logger.info("临时知识库已清除")

    def build_index(self) -> None:
        """
        构建临时知识库索引
        从缓存文件加载节点，创建并持久化向量索引
        支持全量构建和增量更新
        """
        start_time = time.time()
        logger.info(f"开始构建临时库索引，meta缓存：{self.meta_nodes_path}，review缓存：{self.review_nodes_path}")
    
        os.makedirs(self.meta_index_dir, exist_ok=True)
        os.makedirs(self.review_index_dir, exist_ok=True)

        # 检查是否启用增量更新
        if self.enable_incremental_update and \
           os.path.exists(os.path.join(self.meta_index_dir, "index_store.json")) and \
           os.path.exists(os.path.join(self.review_index_dir, "index_store.json")):
            logger.info("启用增量更新模式")
            self.incremental_update()
        else:
            logger.info("启用全量构建模式")
            meta_nodes = load_nodes_from_cache(self.meta_nodes_path)
            review_nodes = load_nodes_from_cache(self.review_nodes_path)

            if meta_nodes:
                from tqdm import tqdm
                with tqdm(total=len(meta_nodes), desc="构建meta索引") as pbar:
                    storage_context_meta = self._create_storage_context()
                    self.meta_index = VectorStoreIndex(
                        nodes=meta_nodes,
                        storage_context=storage_context_meta,
                        embed_model=self.embed_model,
                        show_progress=True
                    )
                    self.meta_index.storage_context.persist(persist_dir=self.meta_index_dir)
                    pbar.update(len(meta_nodes))
                # 更新最后更新时间
                self.last_updated["meta"] = time.time()
            else:
                logger.warning("无meta节点，跳过meta索引构建")

            if review_nodes:
                from tqdm import tqdm
                with tqdm(total=len(review_nodes), desc="构建review索引") as pbar:
                    storage_context_review = self._create_storage_context()
                    self.review_index = VectorStoreIndex(
                        nodes=review_nodes,
                        storage_context=storage_context_review,
                        embed_model=self.embed_model,
                        show_progress=True
                    )
                    self.review_index.storage_context.persist(persist_dir=self.review_index_dir)
                    pbar.update(len(review_nodes))
                # 更新最后更新时间
                self.last_updated["review"] = time.time()
            else:
                logger.warning("无review节点，跳过review索引构建")

            # 保存最后更新时间
            self._save_last_updated()

        end_time = time.time()
        logger.info(f"临时库索引构建完成，耗时：{end_time - start_time:.2f}s")

    def get_retrievers(self) -> Tuple[VectorStoreIndex, VectorStoreIndex, BGERerankerModel]:
        """
        获取临时知识库的检索器
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
            raise RuntimeError(f"临库索引未加载，缺失：{','.join(missing)}")

    def incremental_update(self) -> None:
        """
        增量更新临时知识库索引
        仅更新修改过的节点
        """
        logger.info("执行临时库增量更新...")
        start_time = time.time()

        # 加载现有索引
        if self.meta_index is None and os.path.exists(os.path.join(self.meta_index_dir, "index_store.json")):
            meta_sc = StorageContext.from_defaults(persist_dir=self.meta_index_dir)
            self.meta_index = load_index_from_storage(meta_sc)
            self.meta_index._embed_model = self.embed_model

        if self.review_index is None and os.path.exists(os.path.join(self.review_index_dir, "index_store.json")):
            review_sc = StorageContext.from_defaults(persist_dir=self.review_index_dir)
            self.review_index = load_index_from_storage(review_sc)
            self.review_index._embed_model = self.embed_model

        # 加载最新节点
        current_meta_nodes = load_nodes_from_cache(self.meta_nodes_path)
        current_review_nodes = load_nodes_from_cache(self.review_nodes_path)

        # 比较节点数判断是否有更新
        meta_updated = len(current_meta_nodes) != len(self.meta_nodes)
        review_updated = len(current_review_nodes) != len(self.review_nodes)

        # 更新meta索引
        if meta_updated and self.meta_index:
            logger.info(f"更新meta索引，新增{len(current_meta_nodes) - len(self.meta_nodes)}个节点")
            # 获取新增节点
            new_meta_nodes = [node for node in current_meta_nodes if node.id_ not in [n.id_ for n in self.meta_nodes]]
            if new_meta_nodes:
                self.meta_index.insert_nodes(new_meta_nodes)
                self.meta_index.storage_context.persist(persist_dir=self.meta_index_dir)
                self.meta_nodes = current_meta_nodes
                # 更新最后更新时间
                self.last_updated["meta"] = time.time()
        elif meta_updated:
            logger.warning("meta索引未加载，无法执行增量更新")

        # 更新review索引
        if review_updated and self.review_index:
            logger.info(f"更新review索引，新增{len(current_review_nodes) - len(self.review_nodes)}个节点")
            # 获取新增节点
            new_review_nodes = [node for node in current_review_nodes if node.id_ not in [n.id_ for n in self.review_nodes]]
            if new_review_nodes:
                self.review_index.insert_nodes(new_review_nodes)
                self.review_index.storage_context.persist(persist_dir=self.review_index_dir)
                self.review_nodes = current_review_nodes
                # 更新最后更新时间
                self.last_updated["review"] = time.time()
        elif review_updated:
            logger.warning("review索引未加载，无法执行增量更新")

        # 保存最后更新时间
        self._save_last_updated()

        # 优化索引
        if self.index_optimization.get("enable_after_update", False):
            self.optimize_index()

        end_time = time.time()
        logger.info(f"临时库增量更新完成，耗时：{end_time - start_time:.2f}s")

    def optimize_index(self):
        """
        优化临时知识库索引
        包括索引压缩、碎片整理、清理无效节点等操作
        """
        logger.info("开始优化临时库索引...")
        start_time = time.time()

        # 索引优化配置
        optimize_config = self.index_optimization
        should_compress = optimize_config.get("compress", False)
        compression_level = optimize_config.get("compression_level", 1)
        remove_duplicates = optimize_config.get("remove_duplicates", True)
        min_doc_size = optimize_config.get("min_doc_size", 10)

        # 优化meta索引
        if self.meta_index:
            logger.info("优化meta索引...")
            # 获取当前存储上下文
            meta_sc = self.meta_index.storage_context
            # 获取文档存储
            docstore = meta_sc.docstore
            # 清理无效节点
            valid_docs = {}
            for doc_id, doc in docstore.docs.items():
                # 移除过小的文档
                if len(doc.text.strip()) >= min_doc_size:
                    valid_docs[doc_id] = doc
                else:
                    logger.debug(f"移除过小的meta节点: {doc_id}")

            # 去重
            if remove_duplicates:
                unique_docs = {}
                seen_hashes = set()
                for doc_id, doc in valid_docs.items():
                    from core.utils import hash_text
                    doc_hash = hash_text(doc.text)
                    if doc_hash not in seen_hashes:
                        seen_hashes.add(doc_hash)
                        unique_docs[doc_id] = doc
                    else:
                        logger.debug(f"移除重复的meta节点: {doc_id}")
                valid_docs = unique_docs

            # 更新文档存储
            if len(valid_docs) != len(docstore.docs):
                docstore.docs = valid_docs
                # 重新创建索引
                self.meta_index = VectorStoreIndex.from_documents(
                    documents=list(valid_docs.values()),
                    storage_context=meta_sc,
                    embed_model=self.embed_model,
                )
                # 持久化更新后的索引
                self.meta_index.storage_context.persist(persist_dir=self.meta_index_dir)
                logger.info(f"meta索引优化完成，保留 {len(valid_docs)} 个节点")
            else:
                logger.info("meta索引无需优化")

        # 优化review索引
        if self.review_index:
            logger.info("优化review索引...")
            # 获取当前存储上下文
            review_sc = self.review_index.storage_context
            # 获取文档存储
            docstore = review_sc.docstore
            # 清理无效节点
            valid_docs = {}
            for doc_id, doc in docstore.docs.items():
                # 移除过小的文档
                if len(doc.text.strip()) >= min_doc_size:
                    valid_docs[doc_id] = doc
                else:
                    logger.debug(f"移除过小的review节点: {doc_id}")

            # 去重
            if remove_duplicates:
                unique_docs = {}
                seen_hashes = set()
                for doc_id, doc in valid_docs.items():
                    from core.utils import hash_text
                    doc_hash = hash_text(doc.text)
                    if doc_hash not in seen_hashes:
                        seen_hashes.add(doc_hash)
                        unique_docs[doc_id] = doc
                    else:
                        logger.debug(f"移除重复的review节点: {doc_id}")
                valid_docs = unique_docs

            # 更新文档存储
            if len(valid_docs) != len(docstore.docs):
                docstore.docs = valid_docs
                # 重新创建索引
                self.review_index = VectorStoreIndex.from_documents(
                    documents=list(valid_docs.values()),
                    storage_context=review_sc,
                    embed_model=self.embed_model,
                )
                # 持久化更新后的索引
                self.review_index.storage_context.persist(persist_dir=self.review_index_dir)
                logger.info(f"review索引优化完成，保留 {len(valid_docs)} 个节点")
            else:
                logger.info("review索引无需优化")

        end_time = time.time()
        logger.info(f"临时库索引优化完成，耗时: {end_time - start_time:.2f}秒")

    def clear_cache(self) -> None:
        """
        清除临时知识库缓存
        删除缓存文件和索引目录
        """
        logger.info("清除临时知识库缓存...")
        try:
            # 删除缓存文件
            if os.path.exists(self.meta_nodes_path):
                os.remove(self.meta_nodes_path)
            if os.path.exists(self.review_nodes_path):
                os.remove(self.review_nodes_path)

            # 删除索引目录
            if os.path.exists(self.meta_index_dir):
                shutil.rmtree(self.meta_index_dir)
            if os.path.exists(self.review_index_dir):
                shutil.rmtree(self.review_index_dir)

            # 重置状态
            self.meta_index = None
            self.review_index = None
            self.meta_nodes = []
            self.review_nodes = []
            self.last_updated = {"meta": 0, "review": 0}
            self._save_last_updated()
            logger.info("临时知识库缓存已清除")
        except Exception as e:
            logger.error(f"清除临时知识库缓存失败: {e}")
            raise