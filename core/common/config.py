"""
配置管理模块
统一管理项目配置的加载、验证和访问
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    配置管理器
    负责加载、验证和管理项目配置
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self._config: Optional[Dict[str, Any]] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """加载配置文件"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {self.config_path}")
        except Exception as e:
            logger.error(f"加载配置文件失败：{e}")
            print(f"❌ 加载配置文件失败：{e}")
            self._config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        if self._config is None:
            return default
        return self._config.get(key, default)
    
    def get_required(self, key: str) -> Any:
        """
        获取必需的配置值
        
        Args:
            key: 配置键
            
        Returns:
            配置值
            
        Raises:
            ValueError: 当配置键不存在时
        """
        value = self.get(key)
        if value is None:
            raise ValueError(f"配置缺少必要参数: {key}")
        return value
    
    def validate_required_keys(self, required_keys: list) -> None:
        """
        验证必需的配置键是否存在
        
        Args:
            required_keys: 必需的配置键列表
            
        Raises:
            ValueError: 当缺少必需配置时
        """
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"配置缺少必要参数: {', '.join(missing_keys)}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """获取完整配置字典"""
        return self._config or {}
    
    def reload(self) -> None:
        """重新加载配置文件"""
        self._load_config()


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: str = "config/config.yaml") -> ConfigManager:
    """
    获取全局配置管理器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ConfigManager实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    兼容函数：加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Dict[str, Any]: 配置字典
    """
    return get_config_manager(config_path).config
