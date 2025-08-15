"""
文件操作工具模块
提供文件读写、目录管理等功能
"""

import os
import json
import shutil
import threading
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

_write_lock = threading.Lock()


class FileManager:
    """
    文件管理器
    提供文件操作相关功能
    """
    
    @staticmethod
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
    
    @staticmethod
    def atomic_write(file_path: str, content: Any, mode: str = 'w') -> None:
        """
        原子写入文件
        
        Args:
            file_path: 文件路径
            content: 写入内容
            mode: 写入模式
        """
        with _write_lock:
            try:
                temp_path = f"{file_path}.tmp"
                if isinstance(content, (dict, list)):
                    with open(temp_path, mode, encoding='utf-8') as f:
                        json.dump(content, f, ensure_ascii=False, indent=2)
                else:
                    with open(temp_path, mode, encoding='utf-8') as f:
                        f.write(str(content))
                
                # 原子性重命名
                shutil.move(temp_path, file_path)
                logger.debug(f"原子写入成功: {file_path}")
            except Exception as e:
                logger.error(f"原子写入失败 {file_path}: {e}")
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise
    
    @staticmethod
    def safe_read_json(file_path: str) -> Optional[Dict[str, Any]]:
        """
        安全读取JSON文件
        
        Args:
            file_path: 文件路径
        
        Returns:
            Dict[str, Any] 或 None: JSON内容
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取JSON文件失败 {file_path}: {e}")
            return None
    
    @staticmethod
    def safe_write_json(file_path: str, data: Dict[str, Any]) -> bool:
        """
        安全写入JSON文件
        
        Args:
            file_path: 文件路径
            data: 要写入的数据
        
        Returns:
            bool: 是否成功
        """
        try:
            FileManager.atomic_write(file_path, data)
            return True
        except Exception as e:
            logger.error(f"写入JSON文件失败 {file_path}: {e}")
            return False
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """
        获取文件大小
        
        Args:
            file_path: 文件路径
        
        Returns:
            int: 文件大小（字节）
        """
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0
    
    @staticmethod
    def list_files(directory: str, pattern: str = "*") -> List[str]:
        """
        列出目录中的文件
        
        Args:
            directory: 目录路径
            pattern: 文件模式
        
        Returns:
            List[str]: 文件路径列表
        """
        try:
            if not os.path.exists(directory):
                return []
            
            files = []
            for file in os.listdir(directory):
                if pattern == "*" or file.endswith(pattern):
                    files.append(os.path.join(directory, file))
            return files
        except Exception as e:
            logger.error(f"列出文件失败 {directory}: {e}")
            return []


# 兼容函数
def ensure_dirs(*dirs: str) -> None:
    """兼容函数：确保目录存在"""
    FileManager.ensure_dirs(*dirs)


def atomic_write(file_path: str, content: Any, mode: str = 'w') -> None:
    """兼容函数：原子写入文件"""
    FileManager.atomic_write(file_path, content, mode)
