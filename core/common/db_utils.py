"""
数据库工具模块
提供数据库操作相关功能
"""

import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseUtils:
    """
    数据库工具类
    提供数据库操作相关功能
    """
    
    @staticmethod
    def load_csv_data(file_path: str, encoding: str = 'utf-8') -> Optional[pd.DataFrame]:
        """
        加载CSV数据
        
        Args:
            file_path: CSV文件路径
            encoding: 文件编码
        
        Returns:
            pd.DataFrame 或 None: 数据框
        """
        try:
            if not Path(file_path).exists():
                logger.warning(f"CSV文件不存在: {file_path}")
                return None
            
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"成功加载CSV数据: {file_path}, 行数: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"加载CSV数据失败 {file_path}: {e}")
            return None
    
    @staticmethod
    def save_csv_data(df: pd.DataFrame, file_path: str, encoding: str = 'utf-8') -> bool:
        """
        保存数据到CSV文件
        
        Args:
            df: 数据框
            file_path: 保存路径
            encoding: 文件编码
        
        Returns:
            bool: 是否成功
        """
        try:
            df.to_csv(file_path, index=False, encoding=encoding)
            logger.info(f"成功保存CSV数据: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存CSV数据失败 {file_path}: {e}")
            return False
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        验证数据框是否包含必需的列
        
        Args:
            df: 数据框
            required_columns: 必需的列名列表
        
        Returns:
            bool: 是否有效
        """
        try:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"数据框缺少必需列: {missing_columns}")
                return False
            return True
        except Exception as e:
            logger.error(f"验证数据框失败: {e}")
            return False
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        清理数据框（去除空值、重复值等）
        
        Args:
            df: 原始数据框
        
        Returns:
            pd.DataFrame: 清理后的数据框
        """
        try:
            # 去除完全为空的行
            df = df.dropna(how='all')
            # 去除重复行
            df = df.drop_duplicates()
            # 重置索引
            df = df.reset_index(drop=True)
            logger.info(f"数据框清理完成，剩余行数: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"清理数据框失败: {e}")
            return df


# 兼容函数
def load_csv_data(file_path: str, encoding: str = 'utf-8') -> Optional[pd.DataFrame]:
    """兼容函数：加载CSV数据"""
    return DatabaseUtils.load_csv_data(file_path, encoding)


def save_csv_data(df: pd.DataFrame, file_path: str, encoding: str = 'utf-8') -> bool:
    """兼容函数：保存CSV数据"""
    return DatabaseUtils.save_csv_data(df, file_path, encoding)