import os
import time
import json
import yaml
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from core.main_kb import MainKnowledgeManager
from core.temp_kb import TempKnowledgeManager
from core.llm_service import LLMService
from core.kg_loader import KnowledgeGraphLoader
from llama_index.core.node_parser import SentenceWindowNodeParser
from core.evaluator import get_checked_answer
from core.utils import ensure_dirs, visualize_retrieved_nodes, load_nodes_from_cache, init_nltk, validate_config, format_time

# 兼容函数：获取NodeWithScore或TextNode的文本内容
def _nws_text(nws):
    if hasattr(nws, "get_text"):  # NodeWithScore 新接口
        return nws.get_text().strip()
    n = getattr(nws, "node", None)
    if n and hasattr(n, "get_content"):
        return n.get_content().strip()
    # 兜底
    return str(nws).strip()

# 初始化日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("system.log", encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Dict[str, Any]: 配置字典
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败：{e}")
        print(f"❌ 加载配置文件失败：{e}")
        return {}


def init_services(config: Dict[str, Any]) -> Tuple[LLMService, MainKnowledgeManager, TempKnowledgeManager, SentenceWindowNodeParser, KnowledgeGraphLoader]:
    """
    初始化各项服务
    
    Args:
        config: 配置字典
    
    Returns:
        Tuple[LLMService, MainKnowledgeManager, TempKnowledgeManager, SentenceWindowNodeParser, KnowledgeGraphLoader]: 初始化的服务
    """
    # 初始化LLM服务
    try:
        llm = LLMService(config)
        logger.info("LLM服务初始化成功")
    except Exception as e:
        logger.error(f"LLM服务初始化失败：{e}")
        print(f"❌ LLM服务初始化失败：{e}")
        raise

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 初始化nltk
    nltk_data_path = config.get("nltk_data_path", "")
    if not init_nltk(nltk_data_path):
        logger.warning("nltk初始化不完全，但继续运行")

    # 初始化主知识库
    main_kb = MainKnowledgeManager(config)
    try:
        # 尝试加载已存在的索引
        main_meta_ret, main_review_ret, main_reranker = main_kb.get_retrievers()
        logger.info("主知识库索引加载成功")
    except RuntimeError:
        # 如果索引不存在，构建索引
        logger.info("主知识库索引不存在，开始构建")
        main_kb.build_index()
        main_meta_ret, main_review_ret, main_reranker = main_kb.get_retrievers()
        logger.info("主知识库索引构建成功")
    except Exception as e:
        logger.error(f"主库初始化失败：{e}")
        print(f"❌ 主库初始化失败：{e}")
        main_meta_ret, main_review_ret, main_reranker = None, None, None

    # 初始化临时知识库
    temp_kb = TempKnowledgeManager(config)
    try:
        temp_meta_ret, temp_review_ret, temp_reranker = temp_kb.get_retrievers()
        logger.info("临时知识库索引加载成功")
    except Exception as e:
        logger.warning(f"临库初始化失败：{e}")
        # 尝试构建临时库索引
        try:
            logger.info("开始构建临时库索引")
            temp_kb.build_index()
            temp_meta_ret, temp_review_ret, temp_reranker = temp_kb.get_retrievers()
            logger.info("临时知识库索引构建并加载成功")
        except Exception as build_e:
            logger.error(f"临时库索引构建失败：{build_e}")
            temp_meta_ret, temp_review_ret, temp_reranker = None, None, None

    # 初始化知识图谱加载器
    kg_loader = KnowledgeGraphLoader(config)
    try:
        # 加载知识图谱
        kg_loaded = kg_loader.load_graph(format="gexf")
        if not kg_loaded:
            # 尝试以JSON格式加载
            kg_loaded = kg_loader.load_graph(format="json")
        if kg_loaded:
            # 加载向量存储
            vs_loaded = kg_loader.load_vector_store()
            if vs_loaded:
                logger.info("知识图谱和向量存储加载成功")
            else:
                logger.warning("知识图谱加载成功，但向量存储加载失败")
        else:
            logger.error("知识图谱加载失败")
    except Exception as e:
        logger.error(f"知识图谱初始化失败：{e}")
        print(f"⚠️ 知识图谱初始化失败：{e}")

    node_parser = SentenceWindowNodeParser.from_defaults(window_size=3, window_metadata_key="window")
    logger.info("节点解析器初始化成功")

    return llm, main_kb, temp_kb, node_parser, kg_loader


def load_temp_retrievers(temp_kb: TempKnowledgeManager) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    加载临时库检索器
    
    Args:
        temp_kb: 临时知识库管理器
    
    Returns:
        Tuple[Optional[Any], Optional[Any], Optional[Any]]: 临时库检索器
    """
    try:
        if os.path.exists(os.path.join(temp_kb.meta_index_dir, "index_store.json")) and \
           os.path.exists(os.path.join(temp_kb.review_index_dir, "index_store.json")):
            retrievers = temp_kb.get_retrievers()
            logger.info("临时库检索器加载成功")
            return retrievers
        else:
            logger.warning("临时库索引文件不存在")
            return None, None, None
    except Exception as e:
        logger.error(f"加载临时库检索器失败：{e}")
        return None, None, None


def load_eval_data(eval_data_path: str = "eval_data.json") -> List[Dict[str, Any]]:
    """
    加载评测数据
    
    Args:
        eval_data_path: 评测数据文件路径
    
    Returns:
        List[Dict[str, Any]]: 评测数据列表
    """
    try:
        if os.path.exists(eval_data_path):
            with open(eval_data_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            logger.info(f"成功加载评测数据，共{len(eval_data)}条")
            return eval_data
        else:
            logger.warning(f"评测数据文件不存在：{eval_data_path}")
            return []
    except Exception as e:
        logger.error(f"加载评测数据失败：{e}")
        print(f"⚠️ 加载评测数据失败，创建新文件：{e}")
        return []


def process_command(user_input: str, temp_kb: TempKnowledgeManager) -> bool:
    """
    处理命令行命令
    
    Args:
        user_input: 用户输入
        temp_kb: 临时知识库管理器
    
    Returns:
        bool: 是否继续运行
    """
    if user_input.lower() == "exit":
        logger.info("用户退出程序")
        return False
    elif user_input.lower() == "clear":
        # 清除控制台显示（跨平台兼容）
        os.system('cls' if os.name == 'nt' else 'clear')
        print("✅ 历史已清空")
        logger.info("用户清空历史记录")
        return True
    elif user_input.lower().startswith("upload"):
        path = user_input[7:].strip()
        if os.path.exists(path):
            try:
                # 显示上传进度
                with tqdm(total=100, desc="上传文件") as pbar:
                    pbar.update(30)  # 准备阶段
                    temp_kb.upload_files([path])
                    pbar.update(70)  # 完成上传
                # 反馈解析结果
                meta_nodes = load_nodes_from_cache(temp_kb.meta_nodes_path)
                review_nodes = load_nodes_from_cache(temp_kb.review_nodes_path)
                file_size = os.path.getsize(path)/1024
                logger.info(f"上传成功：文件 {path}，大小 {file_size:.2f}KB，解析出 {len(meta_nodes)} 个meta节点和 {len(review_nodes)} 个review节点")
                print(f"✅ 上传成功：文件类型 {os.path.splitext(path)[1]}，大小 {file_size:.2f}KB")
                print(f"📊 解析结果：meta节点 {len(meta_nodes)} 条，review节点 {len(review_nodes)} 条")
                # 自动触发索引构建
                print("🔄 自动构建临时索引...")
                temp_kb.build_index()
                logger.info("临时索引自动构建完成")
                print("✅ 临时索引自动构建完成")
            except Exception as e:
                logger.error(f"上传失败：{e}")
                print(f"⚠️ 上传失败：{e}")
        else:
            logger.warning(f"上传路径不存在：{path}")
            print("路径不存在")
        return True
    elif user_input.lower() == "remove temp":
        temp_kb.clear()
        logger.info("临时知识已移除")
        print("临时知识已移除")
        return True
    elif user_input.lower() == "build_temp_index":
        try:
            temp_kb.build_index()
            logger.info("临时索引构建完成")
            print("临时索引构建完成")
        except Exception as e:
            logger.error(f"构建临时索引失败：{e}")
            print(f"⚠️ 构建临时索引失败：{e}")
        return True
    return False


def analyze_query(llm: LLMService, user_input: str, chat_history: List[Dict[str, str]], kg_loader: Any = None) -> str:
    """
    分析查询并生成优化的检索查询
    
    Args:
        llm: LLM服务
        user_input: 用户输入
        chat_history: 对话历史
        kg_loader: 知识图谱加载器
    
    Returns:
        str: 优化后的查询
    """
    # 从知识图谱中获取实体信息
    kg_entity_info = ""
    if kg_loader and hasattr(kg_loader, 'G') and kg_loader.G is not None:
        try:
            # 尝试从查询中提取实体
            entity_name = user_input.split('的')[0].strip()
            logger.info(f"尝试从查询中提取实体: {entity_name}")

            # 查找匹配的节点
            matching_nodes = []
            # 先尝试完全匹配
            for node_id in kg_loader.G.nodes:
                if node_id == entity_name:
                    matching_nodes.append(node_id)
                    break
            # 如果没有完全匹配，尝试大小写不敏感的包含匹配
            if not matching_nodes:
                for node_id in kg_loader.G.nodes:
                    if entity_name.lower() in node_id.lower():
                        node_info = kg_loader.get_node_info(node_id)
                        # 优先选择类型为'person'或'movie'的节点
                        if node_info and node_info.get('type') in ['person', 'movie']:
                            matching_nodes.append(node_id)
                            break

            if matching_nodes:
                entity_info_list = []
                for node_id in matching_nodes:
                    node_info = kg_loader.get_node_info(node_id)
                    if node_info:
                        node_type = node_info.get('type', 'unknown')
                        info_str = f"实体 '{node_id}' (类型: {node_type}):"
                        # 添加关键属性
                        for key in ['name', 'year', 'director', 'actor', 'genre']:
                            if key in node_info:
                                info_str += f" {key}: {node_info[key]}"
                        entity_info_list.append(info_str)
                kg_entity_info = "\n".join(entity_info_list)
                logger.info(f"从知识图谱获取到实体信息: {kg_entity_info}")
        except Exception as e:
            logger.error(f"知识图谱实体提取失败：{e}")

    # 构建包含对话历史的消息列表用于查询分析
    system_prompt = "你是一个问题分析专家。请分析用户当前问题和对话历史，生成一个优化后的检索查询，帮助系统找到最相关的电影信息。\n\n优化后的查询应:\n1. 明确提到核心实体(如电影名称、导演、演员等)\n2. 包含关键主题词\n3. 简洁明了，不超过50字\n4. 如果问题中已包含明确的电影名称，请保留并强化相关表述"

    # 如果有知识图谱实体信息，添加到系统提示中
    if kg_entity_info:
        system_prompt += f"\n\n以下是从知识图谱中获取的相关实体信息，可供参考:\n{kg_entity_info}"

    analysis_messages = [{
        "role": "system",
        "content": system_prompt
    }]
    
    # 添加最近的5轮对话历史（10条消息）
    recent_history = chat_history[-10:]
    analysis_messages.extend(recent_history)
    
    # 添加当前问题
    analysis_messages.append({"role": "user", "content": user_input})
    
    # 调用LLM分析问题和历史，生成优化查询
    try:
        analysis_response = llm.call(analysis_messages)
        translated = analysis_response["content"].strip()
        logger.info(f"优化后的检索查询: {translated}")
        print(f"🔍 优化后的检索查询: {translated}")
        return translated
    except Exception as e:
        logger.error(f"查询分析失败，使用原始问题: {e}")
        print(f"⚠️ 查询分析失败，使用原始问题: {e}")
        return user_input


def retrieve_documents(translated: str, main_meta_ret: Any, main_review_ret: Any, temp_meta_ret: Any, temp_review_ret: Any) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """
    检索文档
    
    Args:
        translated: 优化后的查询
        main_meta_ret: 主库元数据检索器
        main_review_ret: 主库评论检索器
        temp_meta_ret: 临时库元数据检索器
        temp_review_ret: 临时库评论检索器
    
    Returns:
        Tuple[List[Any], List[Any], List[Any], List[Any]]: 检索结果
    """
    t0 = time.time()
    meta_nodes = []
    try:
        if main_meta_ret:
            meta_nodes = main_meta_ret.retrieve(translated)
            logger.info(f"主库元数据模糊匹配结果: {len(meta_nodes)} 条")
            print(f"🔍 主库元数据模糊匹配结果: {len(meta_nodes)} 条")
    except Exception as e:
        logger.error(f"主库元数据检索失败：{e}")
        print(f"⚠️ 主库元数据检索失败：{e}")

    t1 = time.time()
    rev_nodes = []
    try:
        if main_review_ret:
            rev_nodes = main_review_ret.retrieve(translated)
            logger.info(f"主库评论模糊匹配结果: {len(rev_nodes)} 条")
            print(f"🔍 主库评论模糊匹配结果: {len(rev_nodes)} 条")
    except Exception as e:
        logger.error(f"主库评论检索失败：{e}")
        print(f"⚠️ 主库评论检索失败：{e}")

    t2 = time.time()
    temp_meta_nodes = []
    try:
        if temp_meta_ret:
            temp_meta_nodes = temp_meta_ret.retrieve(translated)
            logger.info(f"临库元数据模糊匹配结果: {len(temp_meta_nodes)} 条")
    except Exception as e:
        logger.error(f"临库元数据检索失败：{e}")
        print(f"⚠️ 临库元数据检索失败：{e}")

    t3 = time.time()
    temp_review_nodes = []
    try:
        if temp_review_ret:
            temp_review_nodes = temp_review_ret.retrieve(translated)
            logger.info(f"临库评论模糊匹配结果: {len(temp_review_nodes)} 条")
    except Exception as e:
        logger.error(f"临库评论检索失败：{e}")
        print(f"⚠️ 临库评论检索失败：{e}")

    t4 = time.time()
    retrieval_time = format_time(t4 - t0)
    logger.info(f"检索耗时：主库元{format_time(t1-t0)}, 主库评{format_time(t2-t1)}, 临库元{format_time(t3-t2)}, 临库评{format_time(t4-t3)}, 总计{retrieval_time}")
    print(f"检索耗时：主库元{t1-t0:.2f}s, 主库评{t2-t1:.2f}s, 临库元{t3-t2:.2f}s, 临库评{t4-t3:.2f}s")

    return meta_nodes, rev_nodes, temp_meta_nodes, temp_review_nodes


def rerank_and_answer(llm: LLMService, user_input: str, weighted_nodes: List[Tuple[Any, float]], main_reranker: Any, temp_reranker: Any, kg_loader: Any = None) -> Tuple[str, Dict[str, Any]]:
    """
    对检索到的节点进行重排序并生成回答
    
    Args:
        llm: LLM服务
        user_input: 用户输入
        weighted_nodes: 带权重的节点列表
        main_reranker: 主库重排序模型
        temp_reranker: 临时库重排序模型
    
    Returns:
        Tuple[str, Dict[str, Any]]: 生成的回答和详细信息
    """
    # 使用重排序模型处理节点
    try:
        reranked_nodes = []
        if main_reranker and weighted_nodes:
            # 提取节点和权重
            nodes, weights = zip(*weighted_nodes)
            # 将NodeWithScore对象转换为文本内容
            node_texts = [_nws_text(node) for node in nodes]
            # 重排序
            reranked_results = main_reranker.rerank(user_input, node_texts, top_k=10)
            # 应用得分阈值过滤
            filtered_results = [result for result in reranked_results if result['score'] >= main_reranker.score_threshold]
            # 提取索引
            reranked_indices = [result['index'] for result in filtered_results]
            # 获取节点
            reranked_nodes = [nodes[i] for i in reranked_indices]
            logger.info(f"重排序完成，原始结果数: {len(reranked_results)}，过滤后结果数: {len(reranked_nodes)}")
        else:
            # 如果没有重排序模型或没有节点，按权重排序并取前10个
            sorted_nodes = sorted(weighted_nodes, key=lambda x: x[1], reverse=True)
            reranked_nodes = [node for node, _ in sorted_nodes[:10]]
            logger.warning("未使用重排序模型，按权重取前10个节点")
    except Exception as e:
        logger.error(f"重排序失败：{e}")
        # 使用原始节点（取前10个）
        reranked_nodes = [node for node, _ in weighted_nodes[:10]]
        print(f"⚠️ 重排序失败，使用原始节点：{e}")

    # 尝试使用知识图谱增强上下文
    kg_context = ""
    if kg_loader and hasattr(kg_loader, 'G') and kg_loader.G is not None:
        try:
            # 1. 改进实体提取逻辑
            entity_name = user_input.split('的')[0].strip()
            logger.info(f"尝试从查询中提取实体: {entity_name}")

            # 2. 精确查找匹配的节点（优先完全匹配）
            matching_nodes = []
            # 先尝试完全匹配（区分大小写）
            for node_id in kg_loader.G.nodes:
                if node_id == entity_name:
                    matching_nodes.append(node_id)
                    break
            # 如果没有完全匹配，尝试大小写不敏感的包含匹配
            if not matching_nodes:
                for node_id in kg_loader.G.nodes:
                    if entity_name.lower() in node_id.lower():
                        node_info = kg_loader.get_node_info(node_id)
                        # 优先选择类型为'person'的节点（导演、演员等）
                        if node_info and node_info.get('type') == 'person':
                            matching_nodes.append(node_id)
                            break
                        # 如果没有找到人物节点，再添加其他类型
                        elif not matching_nodes:
                            matching_nodes.append(node_id)

            if matching_nodes:
                kg_info = []
                for node_id in matching_nodes:
                    # 获取节点信息
                    node_info = kg_loader.get_node_info(node_id)
                    if node_info:
                        node_type = node_info.get('type', 'unknown')
                        info_str = f"实体 '{node_id}' (类型: {node_type}):\n"
                        for key, value in node_info.items():
                            if key not in ['embedding', 'type']:  # 跳过嵌入向量和类型
                                info_str += f"  {key}: {value}\n"

                        # 3. 改进导演-电影关系查询
                        if node_type == 'person':
                            # 尝试多种可能的关系类型
                            relation_types = ["由...执导", "导演", "执导"]
                            directed_movies = []
                            for rel_type in relation_types:
                                movies = kg_loader.get_related_nodes(node_id, relation_type=rel_type, node_type="movie")
                                directed_movies.extend(movies)
                                # 如果找到足够的电影，就停止尝试其他关系类型
                                if len(directed_movies) >= 5:
                                    break

                            if directed_movies:
                                info_str += "  执导的电影:\n"
                                # 去重并按年份排序（如果有年份信息）
                                unique_movies = {movie_id: movie_info for movie_id, movie_info in directed_movies}
                                sorted_movies = sorted(unique_movies.items(), key=lambda x: x[1].get('year', 0), reverse=True)
                                for movie_id, movie_info in sorted_movies[:5]:  # 限制数量
                                    movie_name = movie_info.get('name', movie_id)
                                    movie_year = movie_info.get('year', 'unknown')
                                    info_str += f"    - {movie_name} ({movie_year})\n"

                        kg_info.append(info_str)

                kg_context = "\n\n".join(kg_info)
                logger.info(f"知识图谱上下文构建完成，长度：{len(kg_context)}字符")
            else:
                # 如果没有找到匹配节点，使用相似性搜索
                similar_nodes = kg_loader.search_similar_nodes(user_input, top_k=3)
                if similar_nodes:
                    kg_info = []
                    for node_id, score in similar_nodes:
                        # 获取节点信息
                        node_info = kg_loader.get_node_info(node_id)
                        if node_info:
                            node_type = node_info.get('type', 'unknown')
                            info_str = f"相似节点 '{node_id}' (相似度: {score:.2f}, 类型: {node_type}):\n"
                            for key, value in node_info.items():
                                if key != 'embedding':  # 跳过嵌入向量
                                    info_str += f"  {key}: {value}\n"
                            # 如果是人物节点，尝试获取其执导的电影
                            if node_type == 'person':
                                relation_types = ["由...执导", "导演", "执导"]
                                directed_movies = []
                                for rel_type in relation_types:
                                    movies = kg_loader.get_related_nodes(node_id, relation_type=rel_type, node_type="movie")
                                    directed_movies.extend(movies)
                                    if len(directed_movies) >= 3:
                                        break
                                if directed_movies:
                                    info_str += "  执导的电影:\n"
                                    for movie_id, movie_info in directed_movies[:3]:
                                        movie_name = movie_info.get('name', movie_id)
                                        movie_year = movie_info.get('year', 'unknown')
                                        info_str += f"    - {movie_name} ({movie_year})\n"
                            kg_info.append(info_str)
                    kg_context = "\n\n".join(kg_info)
                    logger.info(f"知识图谱上下文构建完成，长度：{len(kg_context)}字符")
        except Exception as e:
            logger.error(f"知识图谱增强失败：{e}")

    # 构建最终上下文
    node_context = "\n\n".join([_nws_text(node) for node in reranked_nodes])
    if kg_context:
        context = f"# 知识图谱信息\n{kg_context}\n\n# 文档检索信息\n{node_context}"
    else:
        context = node_context
    logger.info(f"上下文构建完成，长度：{len(context)}字符")

    # 构建提示
    prompt = f"用户问题: {user_input}\n\n基于以下信息回答问题:\n{context}\n\n要求:\n1. 严格基于提供的上下文信息回答，**不得使用任何外部知识**\n2. 仔细分析上下文，确保回答与问题直接相关\n3. 用中文回答，语言流畅自然\n4. 结构清晰，分点说明，使用标题和子标题增强可读性\n5. 对于不确定的信息，明确说明并提供可能的解释\n6. 如果问题无法从提供的信息中回答，明确说明原因\n7. 回答应简洁明了，避免冗余，同时确保信息完整\n8. 直接引用上下文中的具体信息支持你的回答\n9. 总结主要观点，使回答有明确的结论\n10. 再次检查：确保所有回答内容都来自提供的上下文"
    logger.info("提示构建完成")

    # 调用LLM生成回答
    try:
        messages = [
            {"role": "system", "content": "你是一个专业的电影信息问答助手，拥有丰富的电影知识和出色的问答技巧。你的任务是基于提供的上下文信息，准确、全面地回答用户的电影相关问题。回答应具有深度和洞察力，能够综合分析多种信息源，并以清晰、结构化的方式呈现。对于复杂问题，应提供详细解释和依据。"},
            {"role": "user", "content": prompt}
        ]
        t_llm_start = time.time()
        response = llm.call(messages)
        t_llm_end = time.time()
        answer = response["content"].strip()
        llm_time = format_time(t_llm_end - t_llm_start)
        logger.info(f"LLM调用完成，耗时{llm_time}")
        print(f"💡 LLM调用耗时: {t_llm_end - t_llm_start:.2f}s")

        # 增强格式化回答
        formatted_answer = answer
        # 1. 移除多余空行
        formatted_answer = re.sub(r'(\n\n)+', '\n\n', formatted_answer)
        # 2. 优化标题格式（假设标题以数字或符号开头）
        formatted_answer = re.sub(r'^([0-9]+\.|-|\*)\s+', r'\1 ', formatted_answer, flags=re.MULTILINE)
        # 3. 确保段落之间有适当的空行
        formatted_answer = re.sub(r'([^\n])\n([^\n])', r'\1\n\n\2', formatted_answer)
        # 4. 移除首尾空白
        formatted_answer = formatted_answer.strip()

        # 构建返回信息
        answer_info = {
            "answer": formatted_answer,
            "context_length": len(context),
            "node_count": len(reranked_nodes),
            "llm_time": llm_time
        }
        return formatted_answer, answer_info
    except Exception as e:
        logger.error(f"LLM调用失败：{e}")
        print(f"❌ LLM调用失败：{e}")
        return "很抱歉，生成回答时出错。", {"error": str(e)}


def evaluate_and_save(eval_data: List[Dict[str, Any]], user_input: str, answer: str, answer_info: Dict[str, Any], eval_data_path: str = "eval_data.json") -> None:
    """
    评估回答并保存到评测数据
    
    Args:
        eval_data: 评测数据列表
        user_input: 用户输入
        answer: 生成的回答
        answer_info: 回答相关信息
        eval_data_path: 评测数据保存路径
    """
    try:
        # 生成评估结果
        evaluation = get_checked_answer(user_input, answer)
        logger.info(f"评估完成，得分: {evaluation.get('total_score', 0)}")

        # 构建评测数据条目
        # 确保reranked_nodes已定义
        contexts = []
        if 'reranked_nodes' in locals() or 'reranked_nodes' in globals():
            contexts = [_nws_text(node) for node in reranked_nodes]
        
        eval_item = {
            "question": user_input,
            "answer": answer,
            "contexts": contexts,
            "evaluation": evaluation,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "context_length": answer_info.get("context_length", 0),
            "node_count": answer_info.get("node_count", 0),
            "llm_time": answer_info.get("llm_time", "0s")
        }

        # 添加到评测数据
        eval_data.append(eval_item)

        # 保存评测数据
        with open(eval_data_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        logger.info(f"评测数据已保存，共{len(eval_data)}条")
    except Exception as e:
        logger.error(f"评估或保存失败：{e}")
        print(f"⚠️ 评估或保存失败：{e}")


def main():
    """
    主函数入口
    """
    # 1. 加载配置
    config = load_config()
    if not config:
        return
    
    # 2. 验证配置
    if not validate_config(config):
        return
    
    # 3. 初始化服务
    try:
        llm, main_kb, temp_kb, node_parser, kg_loader = init_services(config)
    except Exception as e:
        logger.error(f"服务初始化失败：{e}")
        return

    # 检查知识图谱是否加载成功
    kg_available = hasattr(kg_loader, 'G') and kg_loader.G is not None
    if kg_available:
        print("✅ 知识图谱加载成功，可用于增强回答质量")
    else:
        print("⚠️ 知识图谱未加载或加载失败")
    
    # 加载临时库检索器
    temp_meta_ret, temp_review_ret, temp_reranker = load_temp_retrievers(temp_kb)
    
    # 加载评测数据
    eval_data = load_eval_data()
    
    print(f"临库状态：meta_ret={bool(temp_meta_ret)}, review_ret={bool(temp_review_ret)}")
    print("命令：exit / clear / upload <path> / remove temp / build_temp_index")
    
    chat_history = []
    main_meta_ret, main_review_ret, main_reranker = main_kb.get_retrievers()
    
    while True:
        user_input = input("\n问题：").strip()
        if not user_input:
            continue
        
        # 处理命令
        if process_command(user_input, temp_kb):
            # 重新加载临时库检索器
            temp_meta_ret, temp_review_ret, temp_reranker = load_temp_retrievers(temp_kb)
            continue
        elif user_input.lower() == "exit":
            break
        
        # 记录开始时间
        t_start = time.time()
        
        # 分析查询
        translated = analyze_query(llm, user_input, chat_history, kg_loader)
        
        # 检索文档
        meta_nodes, rev_nodes, temp_meta_nodes, temp_review_nodes = retrieve_documents(
            translated, main_meta_ret, main_review_ret, temp_meta_ret, temp_review_ret
        )
        
        # 可视化召回结果
        visualize_retrieved_nodes(meta_nodes, "主库元数据")
        visualize_retrieved_nodes(rev_nodes, "主库评论")
        visualize_retrieved_nodes(temp_meta_nodes, "临时库元数据")
        visualize_retrieved_nodes(temp_review_nodes, "临时库评论")
        
        # 为不同库的节点添加权重标记
        weighted_nodes = []
        # 主库元数据节点 - 基础权重1.0
        for node in meta_nodes:
            weighted_nodes.append((node, 1.0))

        # 主库评论节点 - 基础权重1.0
        for node in rev_nodes:
            weighted_nodes.append((node, 1.0))

        # 临时库元数据节点 - 基础权重0.8
        for node in temp_meta_nodes:
            weighted_nodes.append((node, 0.8))

        # 临时库评论节点 - 基础权重0.8
        for node in temp_review_nodes:
            weighted_nodes.append((node, 0.8))
        
        # 重排序并生成回答
        answer, answer_info = rerank_and_answer(llm, user_input, weighted_nodes, main_reranker, temp_reranker, kg_loader)
        
        # 记录结束时间
        t_end = time.time()
        total_time = format_time(t_end - t_start)
        logger.info(f"总耗时: {total_time}")
        print(f"⏱️ 总耗时: {t_end - t_start:.2f}s")
        
        # 打印回答
        print("\n✨ 回答:")
        print(answer)
        
        # 更新对话历史
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": answer})
        # 限制历史长度
        if len(chat_history) > 20:  # 保留最近10轮对话
            chat_history = chat_history[-20:]
        
        # 评估并保存
        evaluate_and_save(eval_data, user_input, answer, answer_info)

if __name__ == "__main__":
    main()
