from llama_index.core.schema import Document, TextNode
import os
import json
import hashlib
import pandas as pd
import threading
import re
import csv
import logging
import yaml
from typing import List, Dict, Any, Optional, Tuple, Union
from nltk.tokenize import sent_tokenize

# 初始化日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

_write_lock = threading.Lock()


def ensure_dirs(*dirs: str) -> None:
    """
    确保目录存在
    
    Args:
        *dirs: 目录路径列表
    """
    for d in dirs:
        if d:  # 避免空路径
            try:
                os.makedirs(d, exist_ok=True)
                logger.debug(f"已确保目录存在: {d}")
            except OSError as e:
                logger.error(f"创建目录失败 {d}: {e}")


def hash_text(text: str) -> str:
    """
    计算文本的MD5哈希值
    
    Args:
        text: 输入文本
    
    Returns:
        str: MD5哈希值
    """
    try:
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    except Exception as e:
        logger.error(f"计算文本哈希失败: {e}")
        return ""  # 返回空字符串作为错误情况的默认值


def clean_text(text: Any) -> str:
    """
    清理文本，保留中文和基本格式
    
    Args:
        text: 输入文本，可以是任何类型，会被转换为字符串
    
    Returns:
        str: 清理后的文本（保留大小写）
    """
    try:
        if not isinstance(text, str):
            text = str(text)
        # 保留常见有用符号
        text = re.sub(r"[^\x00-\x7F\u4e00-\u9fa5：，。？！,.?!]+", " ", text)
        # 合并多个空格为一个
        cleaned_text = re.sub(r"\s+", " ", text.strip())
        # 保留大小写
        return cleaned_text
    except Exception as e:
        logger.error(f"清理文本失败: {e}")
        return str(text)  # 出错时返回原始文本的字符串表示


def load_nodes_from_cache(path: Optional[str]) -> List[Dict[str, Any]]:
    """
    从缓存文件加载节点
    
    Args:
        path: 缓存文件路径
    
    Returns:
        List[Dict[str, Any]]: 节点字典列表
    """
    if not path or not os.path.exists(path):
        logger.warning(f"缓存文件不存在: {path}")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        logger.debug(f"从缓存加载节点成功: {path}, 共{len(json_data)}个节点")
        return [d for d in json_data if isinstance(d, dict)]
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"加载缓存节点失败 {path}: {e}")
        return []


def visualize_retrieved_nodes(nodes: List[Any], node_type: str, save_path: Optional[str] = None) -> None:
    """
    可视化显示召回的节点信息
    
    Args:
        nodes: 召回的节点列表
        node_type: 节点类型描述（如"主库元数据"、"主库评论"等）
        save_path: 保存详细信息的文件路径，默认为None（不保存）
    """
    logger.info(f"===== {node_type} 召回结果 =====")
    logger.info(f"召回数量: {len(nodes)}")
    print(f"\n===== {node_type} 召回结果 ======")
    print(f"召回数量: {len(nodes)}")
    
    # 准备详细信息
    details = []
    for i, node in enumerate(nodes[:5]):  # 只显示前5个节点的预览
        try:
            # 尝试获取节点内容
            if hasattr(node, 'node'):
                content = node.node.get_content()
            else:
                content = str(node)
                
            # 截取部分内容作为预览
            preview = content[:100] + "..." if len(content) > 100 else content
            details.append(f"节点 {i+1}:\n{preview}\n")
            
            # 打印预览
            print(f"节点 {i+1}: {preview}")
        except Exception as e:
            error_msg = f"节点 {i+1}: 解析失败 - {e}"
            logger.error(error_msg)
            print(error_msg)
    
    # 如果有更多节点，提示用户
    if len(nodes) > 5:
        more_msg = f"... 以及其他 {len(nodes) - 5} 个节点"
        print(more_msg)
    
    # 保存详细信息到文件
    if save_path:
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(f"===== {node_type} 召回结果 =====\n")
                f.write(f"召回数量: {len(nodes)}\n\n")
                for i, node in enumerate(nodes):
                    try:
                        if hasattr(node, 'node'):
                            content = node.node.get_content()
                        else:
                            content = str(node)
                        f.write(f"节点 {i+1}:\n{content}\n\n")
                    except Exception as e:
                        f.write(f"节点 {i+1}: 解析失败 - {e}\n\n")
            logger.info(f"详细信息已保存到: {save_path}")
            print(f"详细信息已保存到: {save_path}")
        except Exception as e:
            error_msg = f"保存详细信息失败: {e}"
            logger.error(error_msg)
            print(error_msg)


def atomic_write(path: str, data: Any) -> None:
    """
    原子写入数据到文件
    先写入临时文件，成功后再替换原文件，避免文件损坏
    
    Args:
        path: 文件路径
        data: 要写入的数据
    """
    if not path:
        logger.warning("空路径，跳过写入")
        print("⚠️ 空路径，跳过写入")
        return
    with _write_lock:
        # 先写临时文件，成功后再替换，避免文件损坏
        temp_path = f"{path}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, path)  # 原子操作
            logger.debug(f"原子写入成功: {path}")
        except Exception as e:
            logger.error(f"原子写入失败 {path}: {e}")
            print(f"⚠️ 原子写入失败 {path}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.debug(f"已删除临时文件: {temp_path}")
                except Exception as e2:
                    logger.warning(f"删除临时文件失败 {temp_path}: {e2}")


def load_main_nodes_from_source(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    从源文件加载主知识库节点
    
    Args:
        config: 配置字典
    
    Returns:
        tuple: (meta文档列表, review文档列表)
    """
    try:
        meta_dir = config["main_meta_path"]
        review_dir = config["main_review_path"]
    except KeyError as e:
        logger.error(f"配置中缺少必要路径: {e}")
        return [], []

    meta_docs, review_docs = [], []

    def read_txt_files(folder: str) -> List[Dict[str, Any]]:
        """
        读取文件夹中的txt文件
        
        Args:
            folder: 文件夹路径
        
        Returns:
            List[Dict[str, Any]]: 文档列表
        """
        docs = []
        if not folder or not os.path.exists(folder):
            logger.warning(f"文件夹不存在: {folder}")
            return docs
        for fname in os.listdir(folder):
            if fname.endswith(".txt"):
                fpath = os.path.join(folder, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if text:  # 跳过空文件
                        docs.append({
                            "text": text,
                            "metadata": {"filename": fname}
                        })
                        logger.debug(f"已读取文件: {fpath}")
                except Exception as e:
                    logger.error(f"读取主库文件失败 {fpath}: {e}")
                    print(f"⚠️ 读取主库文件失败 {fpath}: {e}")
        logger.info(f"从 {folder} 读取了 {len(docs)} 个文档")
        return docs

    meta_docs = read_txt_files(meta_dir)
    review_docs = read_txt_files(review_dir)
    return meta_docs, review_docs


def auto_classify_and_cache(upload_path: str, meta_cache: str, review_cache: str) -> None:
    """
    自动分类上传的文件并缓存为节点
    支持文件夹和文件路径输入，自动清理数据并分类为meta和review节点
    
    Args:
        upload_path: 上传文件或文件夹路径
        meta_cache: meta节点缓存文件路径
        review_cache: review节点缓存文件路径
    """
    if not upload_path or not os.path.exists(upload_path):
        logger.warning(f"上传路径不存在: {upload_path}")
        print("⚠️ 上传路径不存在")
        return

    # 处理上传路径（支持文件或文件夹）
    file_paths = []
    try:
        if os.path.isfile(upload_path):
            file_paths = [upload_path]
        else:
            for root, _, files in os.walk(upload_path):
                for fname in files:
                    # 支持常见文本格式
                    if fname.lower().endswith((".txt", ".csv", ".md", ".json")):
                        file_paths.append(os.path.join(root, fname))
        logger.info(f"找到 {len(file_paths)} 个文件待处理")
    except Exception as e:
        logger.error(f"处理上传路径失败: {e}")
        print(f"⚠️ 处理上传路径失败: {e}")
        return

    # 创建MovieInfoExtractor实例
    try:
        from core.movie_info_extractor import MovieInfoExtractor  # 延迟导入以避免循环依赖
        extractor = MovieInfoExtractor()
    except ImportError as e:
        logger.error(f"导入MovieInfoExtractor失败: {e}")
        print(f"⚠️ 导入MovieInfoExtractor失败: {e}")
        return
    except Exception as e:
        logger.error(f"创建MovieInfoExtractor实例失败: {e}")
        print(f"⚠️ 创建MovieInfoExtractor实例失败: {e}")
        return

    # 处理文件
    try:
        meta_nodes, review_nodes = extractor.process_files(file_paths)
        logger.info(f"文件处理完成，得到 {len(meta_nodes)} 个meta节点和 {len(review_nodes)} 个review节点")
    except Exception as e:
        logger.error(f"处理文件失败: {e}")
        print(f"⚠️ 处理文件失败: {e}")
        return

    # 保存节点
    try:
        atomic_write(meta_cache, [n.to_dict() for n in meta_nodes])
        atomic_write(review_cache, [n.to_dict() for n in review_nodes])
        logger.info(f"临库缓存完毕：meta {len(meta_nodes)} 条，review {len(review_nodes)} 条")
        print(f"临库缓存完毕：meta {len(meta_nodes)} 条，review {len(review_nodes)} 条")
    except Exception as e:
        logger.error(f"保存节点失败: {e}")
        print(f"⚠️ 保存节点失败: {e}")


def clean_and_extract_text(input_path: str, output_path: str) -> bool:
    """
    清理文本并提取有用信息
    
    Args:
        input_path: 输入文件路径
        output_path: 输出清理后的文件路径
    
    Returns:
        bool: 是否清理成功
    """
    try:
        # 根据文件类型进行不同的清理处理
        if input_path.lower().endswith(".csv"):
            # CSV文件无需提前清理，在处理时再清理
            return True
        elif input_path.lower().endswith(".json"):
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text = json.dumps(data, ensure_ascii=False)
        else:
            # 文本文件
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(input_path, 'r', encoding='latin-1') as f:
                    text = f.read()

        # 使用统一的文本清理函数
        cleaned_text = clean_text(text)

        # 保存清理后的文本
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        logger.info(f"已清理并提取文本: {input_path} -> {output_path}")
        return True
    except Exception as e:
        logger.error(f"清理文件失败 {input_path}: {e}")
        print(f"⚠️ 清理文件失败 {input_path}: {e}")
        return False


def process_movie_csv_data(movies_path: str, file_paths: List[str], meta_cache: str, review_cache: str) -> None:
    """
    处理电影CSV数据并分类为meta和review节点
    
    Args:
        movies_path: 电影CSV文件路径
        file_paths: 要处理的文件路径列表
        meta_cache: meta节点缓存文件路径
        review_cache: review节点缓存文件路径
    """
    meta_nodes = []
    review_nodes = []
    
    # 读取CSV文件并处理
    try:
        with open(movies_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 提取meta信息
                meta_text = f"电影标题: {row.get('title', '')}\n"
                meta_text += f"导演: {row.get('director', '')}\n"
                meta_text += f"主演: {row.get('actors', '')}\n"
                meta_text += f"类型: {row.get('genre', '')}\n"
                meta_text += f"上映日期: {row.get('release_date', '')}\n"
                meta_text += f"评分: {row.get('rating', '')}\n"
                meta_text += f"简介: {row.get('overview', '')}"
                
                # 清理文本
                cleaned_meta = clean_text(meta_text)
                meta_nodes.append({
                    "text": cleaned_meta,
                    "metadata": {
                        "title": row.get('title', ''),
                        "type": "meta"
                    }
                })
        logger.info(f"已处理CSV文件: {movies_path}")
    except Exception as e:
        logger.error(f"处理CSV文件失败 {movies_path}: {e}")
        print(f"⚠️ 处理CSV文件失败 {movies_path}: {e}")

    # 保存节点
    try:
        atomic_write(meta_cache, meta_nodes)
        atomic_write(review_cache, review_nodes)
        logger.info(f"CSV数据缓存完毕：meta {len(meta_nodes)} 条，review {len(review_nodes)} 条")
        print(f"CSV数据缓存完毕：meta {len(meta_nodes)} 条，review {len(review_nodes)} 条")
    except Exception as e:
        logger.error(f"保存CSV节点失败: {e}")
        print(f"⚠️ 保存CSV节点失败: {e}")


def process_general_text_files(file_paths: List[str], meta_cache: str, review_cache: str) -> None:
    """
    处理通用文本文件并分类为meta和review节点
    
    Args:
        file_paths: 要处理的文件路径列表
        meta_cache: meta节点缓存文件路径
        review_cache: review节点缓存文件路径
    """
    meta_nodes = []
    review_nodes = []
    
    if not file_paths:
        logger.warning("没有要处理的文件路径")
        print("⚠️ 没有要处理的文件路径")
        return
    
    try:
        from core.movie_info_extractor import MovieInfoExtractor  # 延迟导入以避免循环依赖
        extractor = MovieInfoExtractor()
    except ImportError as e:
        logger.error(f"导入MovieInfoExtractor失败: {e}")
        print(f"⚠️ 导入MovieInfoExtractor失败: {e}")
        return
    except Exception as e:
        logger.error(f"创建MovieInfoExtractor实例失败: {e}")
        print(f"⚠️ 创建MovieInfoExtractor实例失败: {e}")
        return
    
    # 处理文件
    try:
        meta_nodes, review_nodes = extractor.process_files(file_paths)
        logger.info(f"文件处理完成，得到 {len(meta_nodes)} 个meta节点和 {len(review_nodes)} 个review节点")
    except Exception as e:
        logger.error(f"处理文件失败: {e}")
        print(f"⚠️ 处理文件失败: {e}")
        return
    
    # 保存节点
    try:
        atomic_write(meta_cache, [n.to_dict() for n in meta_nodes])
        atomic_write(review_cache, [n.to_dict() for n in review_nodes])
        logger.info(f"文本文件缓存完毕：meta {len(meta_nodes)} 条，review {len(review_nodes)} 条")
        print(f"文本文件缓存完毕：meta {len(meta_nodes)} 条，review {len(review_nodes)} 条")
    except Exception as e:
        logger.error(f"保存文本文件节点失败: {e}")
        print(f"⚠️ 保存文本文件节点失败: {e}")

# 初始化nltk（如果需要）
# 这个函数可以被其他模块在需要时调用

def init_nltk(nltk_data_path: Optional[str] = None, download: bool = False) -> bool:
    """
    初始化nltk
    
    Args:
        nltk_data_path: nltk数据路径
    
    Returns:
        bool: 是否初始化成功
    """
    try:
        import nltk
        if nltk_data_path and os.path.exists(nltk_data_path):
            nltk.data.path.append(nltk_data_path)
            logger.info(f"已添加nltk数据路径: {nltk_data_path}")
        # 可选：下载必要的nltk资源（默认不下载）
        if download:
            try:
                nltk.download('punkt', quiet=True)
                logger.debug("已下载nltk punkt资源")
            except Exception as e:
                logger.warning(f"下载nltk punkt资源失败: {e}")
        return True
    except Exception as e:
        logger.error(f"nltk初始化失败: {e}")
        print(f"⚠️ nltk初始化失败: {e}")
        return False

# 添加一个辅助函数，用于验证配置的有效性

def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置的有效性
    
    Args:
        config: 配置字典
    
    Returns:
        bool: 配置是否有效
    """
    required_keys = [
        "main_meta_path", "main_review_path", "temp_meta_path", 
        "temp_review_path", "embedding_model_path", "llm_api_url"
    ]
    
    # 检查必要键是否存在
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logger.error(f"配置中缺少必要键: {', '.join(missing_keys)}")
        print(f"❌ 配置中缺少必要键: {', '.join(missing_keys)}")
        return False
    
    # 检查路径是否有效
    path_keys = ["main_meta_path", "main_review_path"]
    for key in path_keys:
        path = config[key]
        if path and not os.path.exists(path):
            logger.warning(f"配置路径不存在: {key} = {path}")
            # 仅当该路径应为目录时创建；这里 main_* 路径为目录，允许创建
            try:
                os.makedirs(path, exist_ok=True)
                logger.info(f"已创建目录: {path}")
            except Exception as e:
                logger.error(f"创建目录失败 {path}: {e}")
                print(f"❌ 创建目录失败 {path}: {e}")
                return False
    
    # 检查LLM配置
    if not config.get("llm_api_url"):
        logger.error("LLM API URL不能为空")
        print("❌ LLM API URL不能为空")
        return False
    
    logger.info("配置验证通过")
    return True

# 添加一个辅助函数，用于格式化时间

def format_time(seconds: float) -> str:
    """
    格式化时间（秒）为人类可读的形式
    
    Args:
        seconds: 时间（秒）
    
    Returns:
        str: 格式化后的时间字符串
    """
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

# 添加一个辅助函数，用于计算列表中元素的相似度

def calculate_similarity(list1: List[Any], list2: List[Any]) -> float:
    """
    计算两个列表的相似度（Jaccard指数）
    
    Args:
        list1: 第一个列表
        list2: 第二个列表
    
    Returns:
        float: 相似度分数（0-1之间）
    """
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0


def split_document_into_blocks(text):
    """优化节点分割策略：优先标签分割，无标签时按句子语义拆分"""
    import nltk
    from nltk.tokenize import sent_tokenize
    import yaml
    import os

    # 读取配置文件
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置nltk数据路径
    nltk_data_path = config.get('nltk_data_path', '')
    if nltk_data_path and os.path.exists(nltk_data_path):
        nltk.data.path.append(nltk_data_path)

    # 确保nltk句子分割模型已下载
    nltk.data.find('tokenizers/punkt')
    
    # 1. 尝试按标签分割（保留原逻辑）
    pattern = r'([\n\r]|^)\s*([\u4e00-\u9fff\w]+)[\s:：]+'
    matches = list(re.finditer(pattern, text))
    if matches:
        blocks = []
        start = 0
        for i, match in enumerate(matches):
            end = match.start()
            if start < end:
                blocks.append(text[start:end].strip())
            if i < len(matches) - 1:
                next_start = matches[i+1].start()
                blocks.append(text[end:next_start].strip())
            else:
                blocks.append(text[end:].strip())
            start = next_start if i < len(matches) - 1 else len(text)
        return [block for block in blocks if block]
    
    # 2. 无标签时按句子分割，合并为~200字符的语义块
    sentences = sent_tokenize(text)
    blocks = []
    current_block = []
    current_length = 0
    for sent in sentences:
        sent_length = len(sent)
        # 若当前块+句子长度超过200，且当前块非空，则生成新块
        if current_length + sent_length > 200 and current_block:
            blocks.append(' '.join(current_block))
            current_block = [sent]
            current_length = sent_length
        else:
            current_block.append(sent)
            current_length += sent_length
    # 添加最后一个块
    if current_block:
        blocks.append(' '.join(current_block))
    return blocks


class MovieInfoExtractor:
    """
    电影信息提取器类，支持中英文文件处理
    能够从多种文件类型中提取电影元数据和评论信息
    """
    def __init__(self):
        # 合并中英文meta和review关键词（去重）
        self.meta_keywords = [
            '电影名', '标题', '评分', '导演', '主演', '类型', '上映日期', '简介', 
            '剧情简介', '片长', '制片国家', '语言',
            'title', 'rating', 'director', 'actor', 'genre', 'release date', 
            'overview', 'synopsis', 'runtime', 'country', 'language'
        ]
        
        self.review_keywords = [
            '影评', '评价', '评论', '观后感', '影评分析', '观影感受', '影片点评',
            'review', 'comment', 'impression', 'movie review', 'film critique', 'analysis'
        ]

    def extract_info_from_text(self, text, file_name=None):
        """
        从文本中提取电影信息
        """
        meta_nodes = []
        review_nodes = []
        cleaned_text = clean_text(text)
        
        # 打印调试信息
        print(f"\n处理文件: {file_name}")
        print(f"原始文本长度: {len(text)}")
        print(f"清理后文本长度: {len(cleaned_text)}")
        print(f"清理后文本前50字符: {cleaned_text[:50]}...")
        
        # 首先检查文件名
        if file_name:
            if 'meta' in file_name.lower():
                print("文件名包含'meta'，分类为meta节点")
                meta_node = Document(text=cleaned_text, metadata={
                    'filename': file_name,
                    'type': 'meta'
                })
                meta_nodes.append(meta_node)
                return meta_nodes, review_nodes
            elif 'review' in file_name.lower():
                print("文件名包含'review'，分类为review节点")
                review_node = Document(text=cleaned_text, metadata={
                    'filename': file_name,
                    'type': 'review'
                })
                review_nodes.append(review_node)
                return meta_nodes, review_nodes
        
        # 按块分割文档以提高分类准确性
        blocks = split_document_into_blocks(cleaned_text)
        print(f"分割后块数量: {len(blocks)}")
        
        # 为每个块创建单独的节点
        for i, block in enumerate(blocks):
            print(f"\n块 {i+1}: {block[:50]}...")
            # 检查块是否包含关键词（不区分大小写）
            is_meta = any(kw.lower() in block.lower() for kw in self.meta_keywords)
            is_review = any(kw.lower() in block.lower() for kw in self.review_keywords)
            
            # 详细调试输出：显示匹配的关键词
            matched_meta_kw = [kw for kw in self.meta_keywords if kw.lower() in block.lower()]
            matched_review_kw = [kw for kw in self.review_keywords if kw.lower() in block.lower()]
            print(f"块 {i+1} 匹配的meta关键词: {matched_meta_kw}")
            print(f"块 {i+1} 匹配的review关键词: {matched_review_kw}")
            
            # 修复标签提取逻辑，同时考虑英文冒号和中文冒号
            tag = None
            if ':' in block:
                tag = block.split(':', 1)[0].strip().lower()
            elif '：' in block:
                tag = block.split('：', 1)[0].strip().lower()
            print(f"块 {i+1} 提取的标签: {tag}")
            
            # 如果块以关键词开头，直接分类
            if tag and tag in [kw.lower() for kw in self.meta_keywords]:
                print(f"块 {i+1} 以meta关键词标签 '{tag}' 开头")
                is_meta = True
            elif tag and tag in [kw.lower() for kw in self.review_keywords]:
                print(f"块 {i+1} 以review关键词标签 '{tag}' 开头")
                is_review = True
            
            # 如果没有检测到关键词，对于短文本我们可以根据是否包含电影相关信息来判断
            if not is_meta and not is_review and len(block) < 500:
                # 检查是否包含电影相关关键词
                movie_related = any(kw.lower() in block.lower() for kw in ['电影', 'film', 'movie'])
                if movie_related:
                    print(f"块 {i+1} 包含电影相关关键词，默认分类为meta节点")
                    is_meta = True
            
            # 强制检查：如果块包含明显meta信息，标记为meta
            meta_indicators = ['电影标题', '导演', '主演', '类型', '上映日期', '评分', '简介']
            has_meta_indicator = any(indicator in block for indicator in meta_indicators)
            if has_meta_indicator:
                print(f"块 {i+1} 包含明显的电影元数据: {[ind for ind in meta_indicators if ind in block]}")
                is_meta = True
            
            # 强制检查：如果块包含"影评"，标记为review
            if '影评' in block:
                print(f"块 {i+1} 包含'影评'关键词")
                is_review = True
            
            if is_meta:
                print(f"块 {i+1} 分类为meta节点")
                meta_node = Document(text=block, metadata={
                    'filename': file_name,
                    'type': 'meta',
                    'tag': tag
                })
                meta_nodes.append(meta_node)
            if is_review:
                print(f"块 {i+1} 分类为review节点")
                review_node = Document(text=block, metadata={
                    'filename': file_name,
                    'type': 'review',
                    'tag': tag
                })
                review_nodes.append(review_node)
            
        print(f"最终分类结果: meta {len(meta_nodes)} 条, review {len(review_nodes)} 条")
        return meta_nodes, review_nodes

    def process_file(self, file_path):
        """
        处理单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            tuple: (meta_nodes, review_nodes) - 元数据节点和评论节点列表
        """
        try:
            file_name = os.path.basename(file_path)
            meta_nodes = []
            review_nodes = []
            
            # 根据文件类型读取内容
            if file_path.lower().endswith(".json"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                text = json.dumps(data, ensure_ascii=False)
                return self.extract_info_from_text(text, file_name)
            elif file_path.lower().endswith(".csv"):
                # 对于CSV文件，使用pandas读取并逐行处理
                df = pd.read_csv(file_path)
                
                # 扩展元数据和评论列匹配规则，提高普遍性
                meta_columns = ['title', 'movie_title', 'film_title', '电影名称', '电影标题',
                               'director', '导演', 'actor', '演员', '主演',
                               'genre', '类型', 'category', 'release_date', '上映日期',
                               'release_year', '年份', 'rating', '评分', 'score',
                               'overview', '简介', '剧情简介', '片长', 'runtime',
                               'country', '国家', 'language', '语言']
                
                review_columns = ['review', 'comment', 'critic_review', 'user_review',
                                 '影评', '评价', '观后感', '影片点评', '电影评论',
                                 'movie_review', 'film_review', '观众评论', '专业影评']
                
                # 智能列匹配 - 支持模糊匹配中和英文混合列名
                matched_meta_cols = []
                matched_review_cols = []
                
                for col in df.columns:
                    col_lower = col.lower()
                    # 元数据列匹配
                    if any(meta_col in col_lower for meta_col in meta_columns):
                        matched_meta_cols.append(col)
                    # 评论列匹配
                    if any(review_col in col_lower for review_col in review_columns):
                        matched_review_cols.append(col)
                
                has_review_col = len(matched_review_cols) > 0
                has_meta_col = len(matched_meta_cols) > 0
                
                # 逐行处理CSV数据
                for idx, row in df.iterrows():
                    # 提取meta信息 - 整合所有匹配的元数据列
                    meta_text = []
                    if has_meta_col:
                        for col in matched_meta_cols:
                            if pd.notna(row[col]):
                                # 标准化字段名（中文）
                                normalized_col = col
                                for eng_col, chn_col in {
                                    'title': '电影标题', 'director': '导演', 'actor': '主演',
                                    'genre': '类型', 'release_date': '上映日期', 'rating': '评分',
                                    'overview': '简介', 'runtime': '片长', 'country': '国家'
                                }.items():
                                    if eng_col in col.lower():
                                        normalized_col = chn_col
                                        break
                                meta_text.append(f"{normalized_col}: {row[col]}")
                    
                    if meta_text:
                        cleaned_meta = clean_text('\n'.join(meta_text))
                        # 智能提取标题（支持多种可能的标题列）
                        title_candidates = [row[col] for col in matched_meta_cols if 'title' in col.lower() or '名称' in col.lower()]
                        title = next((str(t).strip() for t in title_candidates if pd.notna(t) and str(t).strip()), f"Movie_{idx}")
                        
                        meta_node = Document(text=cleaned_meta, metadata={
                            'filename': file_name,
                            'type': 'meta',
                            'title': title
                        })
                        meta_nodes.append(meta_node)
                    
                    # 提取review信息 - 整合所有匹配的评论列
                    review_texts = []
                    if has_review_col:
                        for col in matched_review_cols:
                            if pd.notna(row[col]):
                                review_texts.append(str(row[col]))
                    
                    if review_texts:
                        cleaned_review = clean_text('\n'.join(review_texts))
                        # 复用标题提取逻辑
                        title_candidates = [row[col] for col in matched_meta_cols if 'title' in col.lower() or '名称' in col.lower()]
                        title = next((str(t).strip() for t in title_candidates if pd.notna(t) and str(t).strip()), f"Movie_{idx}")
                        
                        review_node = Document(text=cleaned_review, metadata={
                            'filename': file_name,
                            'type': 'review',
                            'title': title
                        })
                        review_nodes.append(review_node)
                
                print(f"CSV文件处理完成: {file_name}")
                print(f"提取的meta节点数: {len(meta_nodes)}")
                print(f"提取的review节点数: {len(review_nodes)}")
                return meta_nodes, review_nodes
            else:
                # 文本文件
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        text = f.read()
                return self.extract_info_from_text(text, file_name)
        except Exception as e:
            print(f"⚠️ 处理文件失败 {file_path}: {e}")
            return [], []

    def process_files(self, file_paths):
        """
        处理多个文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            tuple: (meta_nodes, review_nodes) - 元数据节点和评论节点列表
        """
        meta_nodes = []
        review_nodes = []
        
        for file_path in file_paths:
            file_meta, file_review = self.process_file(file_path)
            meta_nodes.extend(file_meta)
            review_nodes.extend(file_review)
        
        return meta_nodes, review_nodes