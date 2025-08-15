"""
LLM服务实现模块
提供具体的LLM服务实现
"""

import requests
import time
import re
import json
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar
import logging

from .base import BaseLLMService

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LLMService(BaseLLMService):
    """
    LLM服务实现类
    用于通过API调用大语言模型进行对话补全
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM服务
        
        Args:
            config: 配置字典，包含API URL、模型名称和API密钥
        """
        super().__init__(config)
        
        # 验证配置参数
        if not self.validate_config():
            raise ValueError("LLM配置验证失败")
        
        self.max_retries = config.get("llm_max_retries", 3)
        self.timeout = config.get("llm_timeout", 120)
        self.executor = ThreadPoolExecutor(max_workers=config.get("llm_max_workers", 10))
        logger.info(f"LLM服务初始化成功，模型: {self.model_name}")
    
    def validate_config(self) -> bool:
        """
        验证配置
        
        Returns:
            bool: 配置是否有效
        """
        required_config = ["llm_api_url", "llm_api_key", "llm_model_name"]
        for key in required_config:
            if key not in self.config or not self.config[key]:
                logger.error(f"配置缺少必要参数: {key}")
                return False
        return True
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        生成响应
        
        Args:
            prompt: 输入提示
            **kwargs: 其他参数
        
        Returns:
            str: 生成的响应
        """
        messages = [{"role": "user", "content": prompt}]
        result = self.call(messages, **kwargs)
        return result.get("content", "")
    
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
        # 构建包含上下文的提示
        context_text = "\n".join(context)
        full_prompt = f"上下文信息：\n{context_text}\n\n问题：{prompt}"
        return self.generate_response(full_prompt, **kwargs)
    
    def call(self, messages: List[Union[Dict[str, Any], object]], return_think: bool = False) -> Dict[str, Any]:
        """
        调用LLM进行对话补全
        
        Args:
            messages: 消息列表
            return_think: 是否返回思考过程
        
        Returns:
            dict: 包含回复内容和思考过程（如果return_think=True）的字典
        """
        # 确保messages中的所有内容都是可JSON序列化的
        serializable_messages = []
        for msg in messages:
            # 处理ChatMessage对象
            if hasattr(msg, 'dict'):
                # 如果是Pydantic模型，转换为字典
                msg_dict = msg.dict()
            elif hasattr(msg, '__dict__'):
                # 如果是普通对象，使用其属性
                msg_dict = msg.__dict__
            elif isinstance(msg, dict):
                # 如果已经是字典，直接使用
                msg_dict = msg
            else:
                # 其他情况，转换为字符串
                msg_dict = {'content': str(msg)}

            serializable_msg = {}
            for key, value in msg_dict.items():
                # 确保值是可序列化的
                if isinstance(value, (str, int, float, bool, type(None))):
                    serializable_msg[key] = value
                else:
                    serializable_msg[key] = str(value)
            
            serializable_messages.append(serializable_msg)

        # 兼容基础路径：若 api_url 末尾是 "/v1" 或以 "/v1/" 结尾，则拼接 chat completions 路径
        api_url = self.api_url
        if api_url.rstrip('/') == 'http://10.14.92.66:8080/v1' or api_url.rstrip('/').endswith('/v1'):
            api_url = api_url.rstrip('/') + '/chat/completions'

        payload = {
            "model": self.model_name,
            "messages": serializable_messages,
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                if return_think:
                    # 尝试提取思考过程
                    think_content = self._extract_think_content(content)
                    return {
                        "content": content,
                        "think": think_content
                    }
                else:
                    return {"content": content}
                    
            except requests.exceptions.RequestException as e:
                retry_count += 1
                logger.warning(f"LLM API调用失败，重试 {retry_count}/{self.max_retries}: {e}")
                if retry_count >= self.max_retries:
                    logger.error(f"LLM API调用最终失败: {e}")
                    raise
                time.sleep(1)  # 重试前等待1秒
            except Exception as e:
                logger.error(f"LLM调用出现未知错误: {e}")
                raise
    
    def _extract_think_content(self, content: str) -> str:
        """
        从响应中提取思考过程
        
        Args:
            content: 响应内容
        
        Returns:
            str: 思考过程
        """
        # 简单的思考过程提取逻辑
        think_patterns = [
            r"思考过程[：:](.*?)(?=\n|$)",
            r"思考[：:](.*?)(?=\n|$)",
            r"分析[：:](.*?)(?=\n|$)"
        ]
        
        for pattern in think_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def __del__(self):
        """析构函数，清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
