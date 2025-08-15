"""
文本处理工具模块
提供文本清理、处理和分析功能
"""

import re
import hashlib
import logging
from typing import List, Any, Optional
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    文本处理器
    提供各种文本处理功能
    """
    
    @staticmethod
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
            return ""
    
    @staticmethod
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
            return str(text)
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """
        从文本中提取句子
        
        Args:
            text: 输入文本
        
        Returns:
            List[str]: 句子列表
        """
        try:
            return sent_tokenize(text)
        except Exception as e:
            logger.error(f"提取句子失败: {e}")
            # 简单的句子分割作为备选
            return re.split(r'[。！？.!?]', text)
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        标准化文本（转换为小写，去除多余空格）
        
        Args:
            text: 输入文本
        
        Returns:
            str: 标准化后的文本
        """
        try:
            # 转换为小写
            text = text.lower()
            # 去除多余空格
            text = re.sub(r'\s+', ' ', text.strip())
            return text
        except Exception as e:
            logger.error(f"标准化文本失败: {e}")
            return text
    
    @staticmethod
    def extract_keywords(text: str, min_length: int = 2) -> List[str]:
        """
        从文本中提取关键词
        
        Args:
            text: 输入文本
            min_length: 最小关键词长度
        
        Returns:
            List[str]: 关键词列表
        """
        try:
            # 简单的关键词提取：按空格分割，过滤短词
            words = re.findall(r'\b\w+\b', text.lower())
            keywords = [word for word in words if len(word) >= min_length]
            return list(set(keywords))  # 去重
        except Exception as e:
            logger.error(f"提取关键词失败: {e}")
            return []


# 兼容函数
def hash_text(text: str) -> str:
    """兼容函数：计算文本哈希值"""
    return TextProcessor.hash_text(text)


def clean_text(text: Any) -> str:
    """兼容函数：清理文本"""
    return TextProcessor.clean_text(text)


def extract_sentences(text: str) -> List[str]:
    """兼容函数：提取句子"""
    return TextProcessor.extract_sentences(text)
