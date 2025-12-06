"""
魔搭社区文生图插件 - ModFlux
接入魔搭社区文生图模型，支持LLM调用和命令调用
"""

import asyncio
import aiohttp
import base64
import json
import time
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

# AstrBot相关导入
from astrbot.api.all import *
from astrbot.api.star import Star
from astrbot.api.event import AstrMessageEvent
from astrbot.api.core import Context
from astrbot.api import logger
from astrbot.api import AstrBotConfig


class ModFlux(Star):
    """
    魔搭社区文生图插件
    支持通过命令或LLM智能判断生成图片
    """
    
    def __init__(self, context: Context = None, config: AstrBotConfig = None, **kwargs):
        """
        初始化插件
        
        Args:
            context: AstrBot上下文对象（可选，兼容不同版本）
            config: 插件配置对象（可选）
            **kwargs: 额外的关键字参数（兼容未来版本）
        """
        # 首先调用父类构造函数，确保context正确设置
        if context is None:
            # 某些版本可能不传context
            logger.info("[初始化] 未接收到context参数，使用默认初始化")
            super().__init__()
        else:
            super().__init__(context)
        
        # 使用AstrBot提供的logger接口
        self.logger = logger
        
        # 记录初始化信息
        self.logger.info(f"[初始化] 开始初始化ModFlux插件")
        
        # 初始化配置变量
        try:
            # 优先使用传入的config参数
            if config is not None:
                self.config = config
                self.logger.info(f"[初始化] 配置参数已接收，类型: {type(config)}")
            # 兼容旧版本，可能从context中获取配置
            elif hasattr(context, 'config') and context.config is not None:
                self.config = context.config
                self.logger.info("[初始化] 从context获取配置")
            # 使用默认空配置
            else:
                self.config = {}
                self.logger.info("[初始化] 使用默认空配置")
        except Exception as e:
            self.logger.error(f"[初始化] 配置处理失败: {str(e)}")
            self.config = {}
        
        # 默认配置参数
        self.api_key = ""
        self.model = ""
        self.size = "768x512"
        self.api_url = "https://modelscope.cn/api/v1/"
        self.provider = "ms"
        
        # 智能绘画判断相关配置
        self.paint_probability = 0.3
        self.last_paint_time = 0
        self.min_paint_interval = 300
        
        # LLM智能判断配置
        self.enable_llm_judge = False
        
        # 判断是否绘画的大模型配置
        self.judge_llm_api_url = ""
        self.judge_llm_api_key = ""
        self.judge_llm_model = "gpt-3.5-turbo"
        
        # 生成提示词的大模型配置
        self.prompt_llm_api_url = ""
        self.prompt_llm_api_key = ""
        self.prompt_llm_model = "gpt-3.5-turbo"
        
        # 对话历史缓存（用于基于上下文的提示词生成）
        self.max_cache_size = 10
        
        # 创建数据存储目录
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # 初始化对话缓存（从文件加载）
        self.conversation_cache = self._load_conversation_cache()
        
        # 人物扮演形象配置
        self.character_profile = ""
        
        # 创建临时目录用于存储下载的图片
        self.temp_dir_name = "astrbot_images"
        self.temp_dir = Path(tempfile.gettempdir()) / self.temp_dir_name
        self.temp_dir.mkdir(exist_ok=True)

        # 验证必要配置将在on_config_update中进行
        self.logger.info("[初始化] 魔搭社区文生图插件初始化完成")
    
    def _load_conversation_cache(self) -> list:
        """
        从文件加载对话历史缓存
        
        Returns:
            list: 对话历史缓存列表
        """
        cache_file = self.data_dir / "conversation_cache.json"
        try:
            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                self.logger.info(f"[缓存加载] 成功加载对话缓存，共 {len(cache_data)} 条记录")
                return cache_data
        except Exception as e:
            self.logger.warning(f"[缓存加载] 加载对话缓存失败: {str(e)}")
        
        return []

    def _save_conversation_cache(self):
        """
        保存对话历史缓存到文件
        """
        cache_file = self.data_dir / "conversation_cache.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.conversation_cache, f, ensure_ascii=False, indent=2)
            self.logger.debug("[缓存保存] 对话缓存已保存到文件")
        except Exception as e:
            self.logger.warning(f"[缓存保存] 保存对话缓存失败: {str(e)}")

    def _update_conversation_cache(self, content: str, role: str):
        """
        更新对话历史缓存
        
        Args:
            content: 对话内容
            role: 角色（用户/机器人）
        """
        # 添加新的对话记录
        self.conversation_cache.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # 限制缓存大小，移除最旧的记录
        if len(self.conversation_cache) > self.max_cache_size:
            self.conversation_cache.pop(0)
        
        # 保存更新后的缓存
        self._save_conversation_cache()
    
    def set_character_profile(self, profile: str):
        """
        设置人物扮演形象描述
        
        Args:
            profile: 人物扮演形象描述文本
        """
        self.character_profile = profile
        self.logger.info(f"已设置人物扮演形象：{profile}")

    async def _download_image(self, image_url: str) -> str:
        """
        下载图片到本地，返回本地文件路径
        
        Args:
            image_url: 远程图片URL
            
        Returns:
            str: 本地图片文件路径
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        # 生成唯一的文件名
                        filename = hashlib.md5(image_url.encode()).hexdigest() + ".png"
                        file_path = self.temp_dir / filename
                        
                        # 下载图片数据
                        image_data = await response.read()
                        
                        # 保存到本地文件
                        with open(file_path, "wb") as f:
                            f.write(image_data)
                        
                        self.logger.debug(f"图片下载成功，保存路径: {file_path}")
                        return str(file_path)
                    else:
                        error_msg = f"下载图片失败，HTTP状态码: {response.status}"
                        self.logger.error(error_msg)
                        raise Exception(error_msg)
        except aiohttp.ClientError as e:
            error_msg = f"网络请求错误: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        except IOError as e:
            error_msg = f"文件操作错误: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"图片下载过程中发生未知错误: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    async def _image_to_base64(self, image_url: str) -> str:
        """
        将图片转换为base64编码
        
        Args:
            image_url: 图片URL
            
        Returns:
            str: base64编码的图片数据
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        base64_data = base64.b64encode(image_data).decode('utf-8')
                        self.logger.debug(f"图片转base64成功，数据长度: {len(base64_data)}")
                        return base64_data
                    else:
                        error_msg = f"获取图片数据失败，HTTP状态码: {response.status}"
                        self.logger.error(error_msg)
                        raise Exception(error_msg)
        except aiohttp.ClientError as e:
            error_msg = f"网络请求错误: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"图片转base64过程中发生错误: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def on_config_update(self, new_config: AstrBotConfig):
        """
        配置更新回调 - 当插件配置被更新时调用
        
        Args:
            new_config: 更新后的配置对象（AstrBotConfig）
        """
        self.logger.info(f"[配置更新] 接收到新的配置，类型: {type(new_config)}")
        
        # 处理AstrBotConfig对象，确保能正确提取配置值
        try:
            config_dict = {}
            if isinstance(new_config, dict):
                config_dict = new_config
                self.logger.info("[配置更新] 配置为字典类型")
            elif hasattr(new_config, '__dict__'):
                # 如果是对象，转换为字典
                config_dict = vars(new_config)
                self.logger.info("[配置更新] 配置为对象类型，已转换为字典")
            elif hasattr(new_config, 'get'):
                # 如果已经有get方法，直接使用
                config_dict = new_config
                self.logger.info("[配置更新] 配置已有get方法，直接使用")
            else:
                self.logger.warning(f"[配置更新] 无法识别的配置类型: {type(new_config)}，使用空配置")
            
            # 更新配置字典
            self.config = config_dict if isinstance(config_dict, dict) else {}
            self.logger.info("[配置更新] 配置字典已更新")
            
        except Exception as e:
            self.logger.error(f"[配置更新] 配置处理失败: {str(e)}")
            self.config = {}
        
        # 更新API相关参数
        self.api_key = self.config.get("api_key", "")
        self.model = self.config.get("model", "")
        self.size = self.config.get("size", "768x512")
        self.api_url = self.config.get("api_url", "https://modelscope.cn/api/v1/")
        self.provider = self.config.get("provider", "ms")
        
        # 智能绘画判断相关配置
        self.paint_probability = float(self.config.get("paint_probability", 0.3))
        self.min_paint_interval = int(self.config.get("min_paint_interval", 300))
        
        # LLM智能判断配置
        self.enable_llm_judge = self.config.get("enable_llm_judge", False)
        
        # 判断是否绘画的大模型配置
        self.judge_llm_api_url = self.config.get("judge_llm_api_url", "")
        self.judge_llm_api_key = self.config.get("judge_llm_api_key", "")
        self.judge_llm_model = self.config.get("judge_llm_model", "gpt-3.5-turbo")
        
        # 生成提示词的大模型配置
        self.prompt_llm_api_url = self.config.get("prompt_llm_api_url", "")
        self.prompt_llm_api_key = self.config.get("prompt_llm_api_key", "")
        self.prompt_llm_model = self.config.get("prompt_llm_model", "gpt-3.5-turbo")
        
        # 其他配置
        self.max_cache_size = int(self.config.get("max_cache_size", 10))
        self.character_profile = self.config.get("default_character_profile", "")
        
        self.logger.info(f"[配置更新] 配置更新完成，当前配置项数: {len(self.config)}")

    @filter.event_message_type
    async def on_message(self, event: AstrMessageEvent) -> None:
        """
        监听消息事件 - 用于智能绘画判断
        
        Args:
            event: 消息事件对象
        """
        # 如果未启用LLM智能判断，直接返回
        if not self.enable_llm_judge:
            return
            
        try:
            # 获取消息内容
            message_content = event.get_message_content()
            
            # 更新对话缓存
            self._update_conversation_cache(message_content, "user")
            
            # 检查是否应该生成图片
            should_paint = await self._should_generate_image(message_content)
            
            if should_paint:
                # 生成图片提示词
                prompt = await self._generate_image_prompt(message_content)
                
                # 调用图片生成
                await self._generate_and_send_image(event, prompt)
                
        except Exception as e:
            self.logger.error(f"[消息处理] 处理消息时发生错误: {str(e)}")

    async def _should_generate_image(self, message_content: str) -> bool:
        """
        判断是否应该生成图片
        
        Args:
            message_content: 消息内容
            
        Returns:
            bool: 是否应该生成图片
        """
        current_time = time.time()
        
        # 检查时间间隔
        if current_time - self.last_paint_time < self.min_paint_interval:
            return False
        
        # 简单的关键词判断
        image_keywords = ["图片", "图像", "画", "图", "photo", "image", "picture", "draw", "paint"]
        has_image_keyword = any(keyword in message_content.lower() for keyword in image_keywords)
        
        if not has_image_keyword:
            return False
        
        # 概率判断
        import random
        if random.random() > self.paint_probability:
            return False
        
        return True

    async def _generate_image_prompt(self, message_content: str) -> str:
        """
        生成图片提示词
        
        Args:
            message_content: 消息内容
            
        Returns:
            str: 生成的图片提示词
        """
        # 简单的提示词生成逻辑
        prompt = f"基于以下描述生成图片: {message_content}"
        
        # 如果有角色设定，添加到提示词中
        if self.character_profile:
            prompt = f"角色设定: {self.character_profile}\n{prompt}"
        
        return prompt

    async def _generate_and_send_image(self, event: AstrMessageEvent, prompt: str):
        """
        生成并发送图片
        
        Args:
            event: 消息事件对象
            prompt: 图片提示词
        """
        try:
            # 调用图片生成API
            image_url = await self._generate_image(prompt)
            
            if image_url:
                # 下载图片到本地
                local_path = await self._download_image(image_url)
                
                # 发送图片
                await event.send_message(MessageSegment.image(local_path))
                
                # 更新最后绘画时间
                self.last_paint_time = time.time()
                
                # 更新对话缓存
                self._update_conversation_cache(f"[生成了图片: {prompt}]", "assistant")
                
                self.logger.info(f"[图片生成] 成功生成并发送图片，提示词: {prompt}")
            
        except Exception as e:
            self.logger.error(f"[图片生成] 生成图片失败: {str(e)}")
            await event.send_message(f"图片生成失败: {str(e)}")

    async def _generate_image(self, prompt: str) -> str:
        """
        调用API生成图片
        
        Args:
            prompt: 图片提示词
            
        Returns:
            str: 生成的图片URL
        """
        if not self.api_key:
            raise Exception("API密钥未配置")
        
        if not self.model:
            raise Exception("模型名称未配置")
        
        # 构建请求数据
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "size": self.size,
            "n": 1
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=request_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # 解析响应，获取图片URL
                        if "data" in result and len(result["data"]) > 0:
                            image_url = result["data"][0].get("url", "")
                            if image_url:
                                self.logger.info(f"[API调用] 图片生成成功，URL: {image_url}")
                                return image_url
                            else:
                                raise Exception("API响应中未找到图片URL")
                        else:
                            raise Exception("API响应格式错误")
                    else:
                        error_text = await response.text()
                        raise Exception(f"API调用失败，状态码: {response.status}, 响应: {error_text}")
                        
        except aiohttp.ClientError as e:
            raise Exception(f"网络请求错误: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"JSON解析错误: {str(e)}")
        except Exception as e:
            raise Exception(f"图片生成过程中发生错误: {str(e)}")

    @filter.command("aiimg")
    async def aiimg_command(self, event: AstrMessageEvent, prompt: str):
        """
        AI图片生成命令
        
        Args:
            event: 消息事件对象
            prompt: 图片提示词
        """
        try:
            self.logger.info(f"[命令调用] 收到图片生成请求，提示词: {prompt}")
            
            # 生成图片
            image_url = await self._generate_image(prompt)
            
            if image_url:
                # 下载图片到本地
                local_path = await self._download_image(image_url)
                
                # 发送图片
                await event.send_message(MessageSegment.image(local_path))
                
                self.logger.info(f"[命令调用] 图片生成成功，提示词: {prompt}")
            
        except Exception as e:
            self.logger.error(f"[命令调用] 图片生成失败: {str(e)}")
            await event.send_message(f"图片生成失败: {str(e)}")


# 插件元数据
def create_plugin_metadata():
    """
    创建插件元数据
    
    Returns:
        PluginMetadata: 插件元数据对象
    """
    return PluginMetadata(
        name="ms_ai-g",
        version="1.07",
        description="接入魔搭社区文生图模型。支持LLM调用和命令调用。",
        author="LMG-arch",
        url="https://github.com/LMG-arch/astrbot_plugin_ms_ai-g.git",
        star_cls=ModFlux
    )