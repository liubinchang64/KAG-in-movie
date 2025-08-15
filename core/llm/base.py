"""
LLM基础接口模块
定义LLM服务的抽象基类
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class BaseLLMService(ABC):
    """
    LLM服务基础接口
    定义LLM服务的基本方法和属性
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM服务
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.api_url = config.get("llm_api_url", "")
        self.api_key = config.get("llm_api_key", "")
        self.model_name = config.get("llm_model_name", "")
        self.timeout = config.get("request_timeout", 60)
        self.max_retries = config.get("max_retries", 3)
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        生成响应
        
        Args:
            prompt: 输入提示
            **kwargs: 其他参数
        
        Returns:
            str: 生成的响应
        """
        pass
    
    @abstractmethod
    def generate_response_with_context(self, prompt: str, context: List[str], **kwargs) -> str:
        """
        基于上下文生成响应
        
        Args:
            prompt: 输入提示
            context: 上下文列表
            **kwargs: 其他参数
        
        Returns:
            str: 生成的响应
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        验证配置
        
        Returns:
            bool: 配置是否有效
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "model_name": self.model_name,
            "api_url": self.api_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
