import requests
import time
import re
import json
import requests
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar
import logging

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

T = TypeVar('T')

class LLMService:
    """
    LLM服务类
    用于通过API调用大语言模型进行对话补全
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM服务
        
        Args:
            config: 配置字典，包含API URL、模型名称和API密钥
        """
        # 验证配置参数
        required_config = ["llm_api_url", "llm_api_key", "llm_model_name"]
        for key in required_config:
            if key not in config:
                raise ValueError(f"配置缺少必要参数: {key}")

        self.api_url = config["llm_api_url"]
        self.api_key = config["llm_api_key"]
        self.model_name = config["llm_model_name"]
        self.max_retries = config.get("llm_max_retries", 3)
        self.timeout = config.get("llm_timeout", 120)
        self.executor = ThreadPoolExecutor(max_workers=config.get("llm_max_workers", 10))
        logger.info(f"LLM服务初始化成功，模型: {self.model_name}")

    @property
    def metadata(self):
        """
        获取模型元数据
        
        Returns:
            Meta: 包含模型名称和是否为聊天模型的元数据对象
        """
        class Meta:
            pass

        meta = Meta()
        meta.model_name = self.model_name
        meta.is_chat_model = True  # 补充必要属性
        return meta

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

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model_name,
            "messages": serializable_messages,
            "temperature": 0.5,
            "max_tokens": 10000,
            "stream": False
        }
        t0 = time.time()
        try:
            # 增加超时时间到300秒，并添加重试机制
            max_retries = 3
            retry_count = 0
            response = None
            while retry_count < max_retries:
                try:
                    response = requests.post(
                        f"{self.api_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    break  # 成功获取响应，跳出循环
                except requests.exceptions.Timeout as e:
                    retry_count += 1
                    logger.warning(f"LLM API调用超时，第{retry_count}次重试: {e}")
                    if retry_count >= self.max_retries:
                        logger.error(f"LLM API调用超时，已达到最大重试次数({self.max_retries})")
                        raise
                except requests.exceptions.RequestException as e:
                    logger.error(f"LLM API调用失败: {e}")
                    raise
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API调用失败: {e}")
            return {"content": "", "think": None} if return_think else {"content": ""}

        # 修复：严格检查响应结构
        try:
            data = response.json()
            # 检查必要字段是否存在
            if not isinstance(data, dict) or "choices" not in data:
                raise ValueError("响应缺少'choices'字段")
            if not isinstance(data["choices"], list) or len(data["choices"]) == 0:
                raise ValueError("'choices'为空列表")
            if "message" not in data["choices"][0] or "content" not in data["choices"][0]["message"]:
                raise ValueError("响应缺少'message.content'字段")

            answer = data["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"LLM响应解析失败: {e}")
            return {"content": "", "think": None} if return_think else {"content": ""}

        print(f"LLM耗时：{time.time() - t0:.2f}s")

        # 提取思考过程
        think_text = None
        final_answer = answer.strip()
        try:
            match = re.search(r"<audio>(.*?)<|FunctionCallEnd|>", answer, re.DOTALL)
            if match:
                think_text = match.group(1).strip() if match.group(1) else None
                final_answer = re.sub(r"[SILENT].*?<audio>", "", answer, flags=re.DOTALL).strip()
        except Exception as e:
            logger.error(f"提取思考过程失败: {e}")

        if return_think:
            return {"content": final_answer, "think": think_text}
        else:
            return {"content": final_answer}

    def complete(self, prompt, **kwargs):
        """
        文本补全
        
        Args:
            prompt: 提示文本
            **kwargs: 其他参数
        
        Returns:
            str: 补全后的文本
        """
        messages = [{"role": "user", "content": prompt}]
        return self.call(messages)["content"]

    async def apredict(self, prompt,** kwargs):
        """
        异步文本补全
        
        Args:
            prompt: 提示文本
            **kwargs: 其他参数
        
        Returns:
            str: 补全后的文本
        """
        # 使用线程池执行器确保异步安全
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.complete, prompt)

    def chat(self, messages, **kwargs):
        """
        聊天对话
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
        
        Returns:
            str: 回复内容
        """
        return self.call(messages)["content"]

    async def achat(self, messages,** kwargs):
        """
        异步聊天对话
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
        
        Returns:
            str: 回复内容
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.chat, messages)

    def parallel_call(self, messages_list: List[List[Union[Dict[str, Any], object]]], max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        并行调用LLM进行对话补全
        
        Args:
            messages_list: 消息列表的列表，每个元素是一个消息列表
            max_workers: 最大工作线程数，默认为None(使用类初始化时的设置)
        
        Returns:
            list: 包含每个调用结果的列表
        """
        results = []
        executor = ThreadPoolExecutor(max_workers=max_workers if max_workers is not None else self.executor._max_workers)
        futures = []
        logger.info(f"开始并行调用LLM，共{len(messages_list)}个请求")
        t0 = time.time()

        for messages in messages_list:
            future = executor.submit(self.call, messages)
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"LLM并行调用失败: {e}")
                results.append({"content": ""})

        logger.info(f"LLM并行调用完成，耗时：{time.time() - t0:.2f}s")
        return results