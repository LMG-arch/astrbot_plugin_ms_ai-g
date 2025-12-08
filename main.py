"""
魔搭社区文生图插件 - ModFlux
接入魔搭社区文生图模型，支持LLM调用和命令调用
"""

import asyncio
import aiohttp
import base64
import json
import random
import time
import tempfile
import hashlib
import io
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
import PIL
from PIL import Image

# AstrBot相关导入
from astrbot.api.all import *
from astrbot.api.star import Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.platform import Platform
from astrbot.api import AstrBotConfig


@register("ms_ai-g", "AI绘画插件", "基于魔搭社区的AI绘画生成插件", "2.0")
class ModFlux(Star):
    """
    魔搭社区文生图插件
    支持通过命令或LLM智能判断生成图片
    """
    
    def __init__(self, context, config=None):
        """
        初始化插件
        
        Args:
            context: AstrBot上下文对象
            config: 插件配置对象(AstrBotConfig)，可选
        """
        # 初始化logger
        self.logger = logger
        
        # 记录初始化信息
        self.logger.info(f"[初始化] 开始初始化ModFlux插件，config: {config is not None}")
        
        # 调用父类构造函数
        try:
            super().__init__(context)
            self.logger.info("[初始化] 父类构造函数调用成功")
        except Exception as e:
            self.logger.error(f"[初始化] 调用父类构造函数失败: {str(e)}")
            raise
        
        # 初始化配置变量
        try:
            # 优先使用传入的config参数
            if config is not None:
                self.config = config
                self.logger.info(f"[初始化] 配置参数已接收，类型: {type(config)}")
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
        
        # 智能绘画判断模式
        self.judge_mode = "random"
        
        # 判断是否绘画的大模型配置
        self.judge_llm_api_url = ""
        self.judge_llm_api_key = ""
        self.judge_llm_model = "gpt-3.5-turbo"
        
        # 对话历史缓存（用于基于上下文的提示词生成）
        self.max_cache_size = 10
        
        # 活动状态管理
        self.current_activity = "none"
        self.activity_history = []
        self.activity_images = {
            "none": "A person sitting at a desk, working on a computer, modern office, soft lighting",
            "experiment": "A scientist working in a laboratory, surrounded by test tubes and equipment, bright overhead lighting",
            "shopping": "A busy street scene with shops and people walking around, vibrant colors, daytime",
            "cooking": "A kitchen scene with someone cooking, warm lighting, homely atmosphere",
            "reading": "A person reading a book in a cozy corner, soft lighting, comfortable chair",
            "writing": "A person writing at a desk, papers scattered around, natural lighting from a window",
            "hiking": "A scenic mountain trail with a hiker, beautiful landscape, sunny day",
            "painting": "An artist painting on a canvas, colorful art supplies, studio lighting"
        }
        
        # 创建数据存储目录
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # 初始化对话缓存（从文件加载）
        self.conversation_cache = self._load_conversation_cache()
        

        
        # 临时目录
        self.temp_dir_name = "astrbot_images"
        self.temp_dir = Path(tempfile.gettempdir()) / self.temp_dir_name
        self.temp_dir.mkdir(exist_ok=True)
        
        # 插件初始化完成
        self.logger.info("[初始化] ModFlux插件初始化完成")
        
        # 如果有配置，立即更新
        if self.config:
            self.on_config_update(self.config)
    
    def _load_conversation_cache(self) -> List[Dict]:
        """从文件加载对话缓存"""
        cache_file = self.data_dir / "conversation_cache.json"
        try:
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"[缓存] 加载对话缓存失败: {str(e)}")
        return []
    
    def _save_conversation_cache(self):
        """保存对话缓存到文件"""
        cache_file = self.data_dir / "conversation_cache.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"[缓存] 保存对话缓存失败: {str(e)}")
    
    def add_to_conversation_cache(self, user_id: str, user_message: str, bot_response: str):
        """添加对话到缓存"""
        # 创建新对话记录
        conversation = {
            "user_id": user_id,
            "user_message": user_message,
            "bot_response": bot_response,
            "timestamp": time.time()
        }
        
        # 添加到缓存
        self.conversation_cache.append(conversation)
        
        # 限制缓存大小
        if len(self.conversation_cache) > self.max_cache_size:
            self.conversation_cache = self.conversation_cache[-self.max_cache_size:]
        
        # 保存到文件
        self._save_conversation_cache()
    

    
    async def image_to_base64(self, image_url: str) -> str:
        """
        将图片URL转换为base64编码
        
        Args:
            image_url: 图片URL
            
        Returns:
            图片的base64编码
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

    def on_config_update(self, new_config):
        """
        配置更新回调 - 当插件配置被更新时调用
        
        Args:
            new_config: 更新后的配置对象（可能是AstrBotConfig、dict或其他类型）
        """
        self.logger.info(f"[配置更新] 接收到新的配置，类型: {type(new_config)}")
        
        # 处理不同类型的配置对象，确保能正确提取配置值
        try:
            # AstrBotConfig继承自Dict，可以直接使用
            if hasattr(new_config, 'get') and callable(new_config.get):
                self.config = new_config
                self.logger.info("[配置更新] 配置为AstrBotConfig或字典类型，直接使用")
            elif isinstance(new_config, dict):
                self.config = new_config
                self.logger.info("[配置更新] 配置为字典类型")
            elif hasattr(new_config, '__dict__'):
                # 如果是对象，转换为字典
                self.config = vars(new_config)
                self.logger.info("[配置更新] 配置为对象类型，已转换为字典")
            else:
                self.logger.warning(f"[配置更新] 无法识别的配置类型: {type(new_config)}，使用空配置")
                self.config = {}
            
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
        
        # 绘画判断模式配置
        self.judge_mode = self.config.get("judge_mode", "random")
        # 当配置了LLM API但模式不是llm时，自动切换到llm模式
        if (self.judge_llm_api_key and self.judge_llm_api_url) and self.judge_mode != "llm":
            self.judge_mode = "llm"
            self.logger.info("[配置更新] 已配置LLM API，自动切换到LLM判断模式")
        
        # 判断是否绘画的大模型配置
        self.judge_llm_api_url = self.config.get("judge_llm_api_url", "")
        self.judge_llm_api_key = self.config.get("judge_llm_api_key", "")
        self.judge_llm_model = self.config.get("judge_llm_model", "gpt-3.5-turbo")
        
        # 其他配置
        self.max_cache_size = int(self.config.get("max_cache_size", 10))
        
        self.logger.info(f"[配置更新] 配置更新完成，当前配置项数: {len(self.config)}")

    async def initialize(self):
        # 解决参数不匹配问题：将注册表中的未绑定函数更新为绑定方法
        from astrbot.core.star.star_handler import star_handlers_registry
        
        # 获取当前模块名和方法名
        module_name = self.__class__.__module__
        method_name = "on_message"
        handler_full_name = f"{module_name}_{method_name}"
        
        # 查找注册表中的处理函数
        handler_metadata = star_handlers_registry.get_handler_by_full_name(handler_full_name)
        if handler_metadata:
            # 更新为绑定到当前实例的方法
            handler_metadata.handler = self.on_message
            self.logger.info(f"[ms_ai_g] 已将on_message方法更新为绑定版本")
        
        self.logger.info(f"[初始化] ModFlux插件初始化完成")


    async def generate_image(self, prompt: str, user_id: Optional[str] = None) -> Optional[str]:
        """
        生成图片
        
        Args:
            prompt: 图片生成提示词
            user_id: 用户ID（用于记录）
            
        Returns:
            生成的图片URL，失败返回None
        """
        # 检查API密钥
        if not self.api_key:
            self.logger.error("[图片生成] API密钥未配置")
            return None
        
        # 移除可能存在的角色设定（避免重复添加）
        if "角色设定:" in prompt:
            prompt = prompt.split("角色设定:", 1)[1].strip()
            if "\n" in prompt:
                prompt = prompt.split("\n", 1)[1].strip()
        
        # 不再自动添加角色设定到提示词
            
        # 根据provider或base_url选择不同的API格式
        if self.provider == "openai" or self.provider == "oa" or "api-inference.modelscope.cn" in self.api_url:
            # OpenAI API格式（适用于api-inference.modelscope.cn和显式设置的openai provider）
            base_url = self.api_url.rstrip('/')
            # 检查base_url是否已经包含/v1/，避免重复添加
            if base_url.endswith('/v1'):
                url = f"{base_url}/images/generations"
            else:
                url = f"{base_url}/v1/images/generations"
            
            # 添加日志查看实际构建的URL
            self.logger.info(f"[图片生成] 构建的API URL: {url}")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model or "AI-ModelScope/stable-diffusion-xl-base-1.0",
                "prompt": prompt,
                "n": 1,
                "size": self.size or "1024x1024",
                "response_format": "url"
            }
        else:
            # 默认使用ModelScope API格式
            model_id = self.model or "AI-ModelScope/stable-diffusion-xl-base-1.0"
            
            # 构建正确的API URL
            base_url = self.api_url.rstrip('/')
            
            # 处理ModelScope API URL的不同格式
            if "/infer" in base_url:
                # 如果URL已经包含/infer，则直接使用
                url = base_url
            else:
                # 检查是否已经包含完整的路径结构
                if "v1/models" in base_url and "/" not in base_url.split("v1/models/")[1]:
                    # 格式如: https://api-inference.modelscope.cn/v1/models
                    url = f"{base_url}/{model_id}/infer"
                elif "/models" in base_url and "/" not in base_url.split("/models/")[1]:
                    # 格式如: https://api-inference.modelscope.cn/models
                    url = f"{base_url}/{model_id}/infer"
                elif base_url == "https://modelscope.cn/api/v1":
                    # 默认ModelScope URL格式
                    url = f"{base_url}/models/{model_id}/infer"
                elif "api-inference.modelscope.cn" in base_url:
                    # api-inference.modelscope.cn格式不需要/v1/前缀
                    url = f"{base_url}/models/{model_id}/infer"
                elif not base_url.endswith("/infer"):
                    # 如果不包含/infer，确保添加正确的路径
                    if "/v1/" in base_url:
                        if "/models/" in base_url:
                            url = f"{base_url}/infer"
                        else:
                            url = f"{base_url}/models/{model_id}/infer"
                    else:
                        # 只有非api-inference域名才添加/v1/前缀
                        if "api-inference.modelscope.cn" not in base_url:
                            url = f"{base_url}/v1/models/{model_id}/infer"
                        else:
                            url = f"{base_url}/models/{model_id}/infer"
                
            # 添加日志查看实际构建的URL
            self.logger.info(f"[图片生成] 构建的API URL: {url}")
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "input": {
                    "prompt": prompt,
                    "negative_prompt": "low quality, worst quality, blurry, watermark, signature",
                    "width": int(self.size.split("x")[0]) if "x" in self.size else 768,
                    "height": int(self.size.split("x")[1]) if "x" in self.size else 512,
                    "steps": 30,
                    "guidance_scale": 7.5,
                    "num_images": 1
                }
            }
        
        try:
            self.logger.info(f"[图片生成] 开始生成图片，提示词: {prompt[:50]}...")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # 根据provider或base_url解析不同的响应格式
                        if self.provider == "openai" or self.provider == "oa" or "api-inference.modelscope.cn" in self.api_url:
                            # OpenAI响应格式（适用于api-inference.modelscope.cn和显式设置的openai provider）
                            # 记录完整响应以便调试
                            self.logger.info(f"[图片生成] API响应内容: {json.dumps(result, ensure_ascii=False)}")
                            
                            # 首先尝试OpenAI标准格式
                            data_list = result.get("data", [])
                            if data_list:
                                image_url = data_list[0].get("url")
                                if image_url:
                                    self.logger.info(f"[图片生成] 图片生成成功，URL: {image_url}")
                                    return image_url
                                else:
                                    self.logger.error("[图片生成] OpenAI响应中未找到图片URL")
                            else:
                                # 尝试ModelScope API的兼容格式
                                self.logger.info("[图片生成] 尝试使用ModelScope兼容格式解析响应")
                                output = result.get("output")
                                if output:
                                    if isinstance(output, list) and len(output) > 0:
                                        image_url = output[0].get("url")
                                        if image_url:
                                            self.logger.info(f"[图片生成] 图片生成成功，URL: {image_url}")
                                            return image_url
                                        else:
                                            self.logger.error("[图片生成] ModelScope兼容响应中未找到图片URL")
                                    elif isinstance(output, dict):
                                        image_url = output.get("url")
                                        if image_url:
                                            self.logger.info(f"[图片生成] 图片生成成功，URL: {image_url}")
                                            return image_url
                                        else:
                                            # 检查是否包含images列表
                                            images = output.get("images", [])
                                            if images and isinstance(images, list):
                                                if isinstance(images[0], dict) and "url" in images[0]:
                                                    image_url = images[0].get("url")
                                                    if image_url:
                                                        self.logger.info(f"[图片生成] 图片生成成功，URL: {image_url}")
                                                        return image_url
                                                elif isinstance(images[0], str):
                                                    # 如果是base64字符串，需要保存为文件
                                                    try:
                                                        # 解码base64字符串
                                                        if images[0].startswith("data:image/"):
                                                            # 移除数据URL前缀
                                                            base64_data = images[0].split(",")[1]
                                                        else:
                                                            base64_data = images[0]
                                                             
                                                        image_data = base64.b64decode(base64_data)
                                                         
                                                        # 生成文件名
                                                        timestamp = int(time.time())
                                                        file_name = f"img_{timestamp}.jpg"
                                                        file_path = self.temp_dir / file_name
                                                         
                                                        # 保存图片
                                                        img = Image.open(io.BytesIO(image_data))  # type: ignore[attr-defined]
                                                        img.save(file_path)
                                                        self.logger.info(f"[图片生成] 图片生成成功，已保存到本地: {file_path}")
                                                        return str(file_path)
                                                    except Exception as e:
                                                        self.logger.error(f"[图片生成] 处理base64图片数据失败: {str(e)}")
                                    else:
                                        self.logger.error(f"[图片生成] 响应output格式不匹配: {output}")
                                else:
                                    # 尝试直接从根级别查找images字段
                                    images = result.get("images", [])
                                    if images:
                                        self.logger.info(f"[图片生成] 从根级别找到images字段: {images}")
                                        if isinstance(images[0], dict) and "url" in images[0]:
                                            image_url = images[0].get("url")
                                            if image_url:
                                                self.logger.info(f"[图片生成] 图片生成成功，URL: {image_url}")
                                                return image_url
                                        elif isinstance(images[0], str):
                                            # 直接返回图片URL
                                            self.logger.info(f"[图片生成] 图片生成成功，URL: {images[0]}")
                                            return images[0]
                                    
                                    self.logger.error("[图片生成] 无法解析API响应格式")
                        else:
                            # 默认ModelScope响应格式
                            # ModelScope的响应通常包含output字段
                            self.logger.info(f"[图片生成] ModelScope API响应: {result}")
                            # ModelScope的响应格式应该包含output字段
                            output = result.get("output")
                            if output:
                                # 检查output是否是列表
                                if isinstance(output, list) and len(output) > 0:
                                    image_url = output[0].get("url")
                                    if image_url:
                                        self.logger.info(f"[图片生成] 图片生成成功，URL: {image_url}")
                                        return image_url
                                    else:
                                        self.logger.error("[图片生成] 响应output中未找到图片URL")
                                elif isinstance(output, dict):
                                    # 检查是否直接包含url
                                    image_url = output.get("url")
                                    if image_url:
                                        self.logger.info(f"[图片生成] 图片生成成功，URL: {image_url}")
                                        return image_url
                                    # 或者检查是否包含images列表
                                    images = output.get("images", [])
                                    if images and isinstance(images, list):
                                        if isinstance(images[0], dict) and "url" in images[0]:
                                            image_url = images[0].get("url")
                                            if image_url:
                                                self.logger.info(f"[图片生成] 图片生成成功，URL: {image_url}")
                                                return image_url
                                        elif isinstance(images[0], str):
                                            # 如果是base64字符串，需要保存为文件
                                            
                                            try:
                                                # 解码base64字符串
                                                if images[0].startswith("data:image/"):
                                                    # 移除数据URL前缀
                                                    base64_data = images[0].split(",")[1]
                                                else:
                                                    base64_data = images[0]
                                                    
                                                image_data = base64.b64decode(base64_data)
                                                
                                                # 生成文件名
                                                timestamp = int(time.time())
                                                file_name = f"img_{timestamp}.jpg"
                                                file_path = self.temp_dir / file_name
                                                
                                                # 保存图片
                                                img = Image.open(io.BytesIO(image_data))  # type: ignore[attr-defined]
                                                img.save(file_path)
                                                self.logger.info(f"[图片生成] 图片生成成功，已保存到本地: {file_path}")
                                                return str(file_path)
                                            except Exception as e:
                                                self.logger.error(f"[图片生成] 处理base64图片数据失败: {str(e)}")
                                    else:
                                        self.logger.error(f"[图片生成] 响应output格式不匹配: {output}")
                                else:
                                    self.logger.error(f"[图片生成] 响应output格式不匹配: {output}")
                            else:
                                # 尝试兼容旧格式
                                images = result.get("images", [])
                                if images:
                                    if isinstance(images[0], dict) and "url" in images[0]:
                                        image_url = images[0].get("url")
                                        if image_url:
                                            self.logger.info(f"[图片生成] 图片生成成功（旧格式），URL: {image_url}")
                                            return image_url
                                    elif isinstance(images[0], str):
                                        # 如果是base64字符串，需要保存为文件
                                        
                                        try:
                                            # 解码base64字符串
                                            if images[0].startswith("data:image/"):
                                                # 移除数据URL前缀
                                                base64_data = images[0].split(",")[1]
                                            else:
                                                base64_data = images[0]
                                                
                                            image_data = base64.b64decode(base64_data)
                                            
                                            # 生成文件名
                                            timestamp = int(time.time())
                                            file_name = f"img_{timestamp}.jpg"
                                            file_path = self.temp_dir / file_name
                                            
                                            # 保存图片
                                            img = Image.open(io.BytesIO(image_data))  # type: ignore[attr-defined]
                                            img.save(file_path)
                                            self.logger.info(f"[图片生成] 图片生成成功（旧格式），已保存到本地: {file_path}")
                                            return str(file_path)
                                        except Exception as e:
                                            self.logger.error(f"[图片生成] 处理base64图片数据失败（旧格式）: {str(e)}")
                                else:
                                    self.logger.error("[图片生成] 响应中未找到图片数据")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"[图片生成] API请求失败，状态码: {response.status}, 错误: {error_text}")
        except Exception as e:
            self.logger.error(f"[图片生成] 生成图片时发生错误: {str(e)}")
        
        return None
    
    async def download_image(self, image_url: str) -> Optional[str]:
        """
        下载图片到本地
        
        Args:
            image_url: 图片URL
            
        Returns:
            本地图片路径，失败返回None
        """
        try:
            # 生成文件名
            timestamp = int(time.time())
            hash_obj = hashlib.md5(image_url.encode())
            file_name = f"img_{timestamp}_{hash_obj.hexdigest()[:8]}.jpg"
            file_path = self.temp_dir / file_name
            
            # 下载图片
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        with open(file_path, 'wb') as f:
                            f.write(await response.read())
                        self.logger.info(f"[图片下载] 图片下载成功，路径: {file_path}")
                        return str(file_path)
                    else:
                        self.logger.error(f"[图片下载] 下载失败，状态码: {response.status}")
        except Exception as e:
            self.logger.error(f"[图片下载] 下载图片时发生错误: {str(e)}")
        
        return None
    
    async def should_paint(self, message: str, user_id: str, event: AstrMessageEvent) -> bool:
        """
        判断用户是否希望进行绘画
        
        Args:
            message: 用户消息
            user_id: 用户ID
            event: 消息事件对象
            
        Returns:
            请认真思考这句对话用户是否希望进行绘画
        """
        # 定义绘画关键词
        paint_keywords = ["看看照片", "看看", "发张照片","拍张照片", "发张图", "生成图片", "画一张", "画个", "想看照片", "想看"]
        
        # 检查是否包含绘画关键词
        detected_keywords = []
        for keyword in paint_keywords:
            if keyword in message:
                detected_keywords.append(keyword)
        
        if detected_keywords:
            # 如果检测到关键词，直接判断为需要绘画
            self.logger.info(f"[智能判断] 检测到绘画关键词: {', '.join(detected_keywords)}，直接判断为需要绘画")
            return True
        
        # 检查是否在最小间隔时间内
        current_time = time.time()
        
        # 检查是否是活动询问
        activity_query_phrases = ["在干嘛", "在做什么", "你在干什么", "你在做什么", "你在忙什么", "在忙什么"]
        if any(phrase in message for phrase in activity_query_phrases):
            # 如果是活动询问，不受最小间隔时间限制，直接判断为需要绘画
            self.logger.info(f"[智能判断] 检测到活动询问: {message}，不受间隔限制，直接判断为需要绘画")
            return True
        
        # 非活动询问，检查最小间隔时间
        if current_time - self.last_paint_time < self.min_paint_interval:
            self.logger.info("[智能判断] 在最小间隔时间内，跳过绘画")
            return False
        
        # 检查判断模式配置
        judge_mode = self.config.get("judge_mode", "random")
        
        # 随机模式
        if judge_mode == "random":
            paint_probability = self.config.get("paint_probability", 0.3)
            if random.random() < paint_probability:
                self.logger.info(f"[随机判断] 随机触发绘画，概率: {paint_probability}")
                return True
            else:
                self.logger.info(f"[随机判断] 未触发绘画，概率: {paint_probability}")
                return False
        
        # LLM判断模式 - 使用AstrBot提供的统一大模型接口
        elif judge_mode == "llm":
            try:
                self.logger.info("[智能判断] 使用AstrBot统一大模型接口进行判断")
                
                # 获取系统消息历史
                try:
                    # 从event中获取platform_id和user_id
                    platform_id = event.platform_meta.id
                    user_id = event.unified_msg_origin
                    
                    system_messages = await self.context.message_history_manager.get(
                        platform_id=platform_id,
                        user_id=user_id,
                        page=1,
                        page_size=5
                    )
                    
                    conversation_text = ""
                    if system_messages:
                        for msg in system_messages:
                            # 解析消息内容
                            if msg.content and isinstance(msg.content, dict):
                                content = msg.content.get('message_str', '')
                                sender_name = msg.sender_name or ''
                                role = "user" if sender_name != "AstrBot" else "assistant"
                                conversation_text += f"{role}: {content}\n"
                        self.logger.info(f"[智能判断] 已获取 {len(system_messages)} 条系统消息历史")
                    else:
                        self.logger.info("[智能判断] 未获取到系统消息历史")
                except Exception as e:
                    self.logger.error(f"[智能判断] 获取消息历史时发生错误: {str(e)}")
                    conversation_text = ""
                
                # 构建判断请求
                judge_prompt = f"""
你现在需要扮演一个绘画请求判断专家，请仔细分析用户的最新对话，判断用户是否希望进行绘画生成。

用户最新消息：{message}

对话历史：
{conversation_text}

请你只需要输出 "yes" 或 "no"，表示是否需要进行绘画生成，不要输出其他任何内容。
如果用户的消息中有绘画相关的请求，比如"画一张"、"生成图片"、"发张图"、"我要瞧瞧"、"我想看看"、"我想看"、"给我看"、"展示"等，或者用户询问"在干嘛"、"在做什么"、"你在干什么"、"你在做什么"、"你在忙什么"等关于当前活动的问题，就输出 "yes"，否则输出 "no"。
"""
                
                # 使用AstrBot提供的统一大模型接口
                try:
                    umo = event.unified_msg_origin
                    provider_id = await self.context.get_current_chat_provider_id(umo=umo)
                    
                    self.logger.info(f"[智能判断] 获取到当前会话的模型提供商: {provider_id}")
                    
                    llm_resp = await self.context.llm_generate(
                        chat_provider_id=provider_id,
                        prompt=judge_prompt
                    )
                    
                    if llm_resp and llm_resp.completion_text:
                        judgment = llm_resp.completion_text.strip().lower()
                        self.logger.info(f"[智能判断] 大模型判断结果: {judgment}")
                        return judgment == "yes"
                    else:
                        self.logger.error(f"[智能判断] 大模型响应为空: {llm_resp}")
                        # 响应为空，回退到随机判断模式
                        self.logger.info("[智能判断] 响应为空，回退到随机判断模式")
                        paint_probability = self.config.get("paint_probability", 0.3)
                        if random.random() < paint_probability:
                            self.logger.info(f"[随机判断(回退)] 随机触发绘画，概率: {paint_probability}")
                            return True
                        else:
                            self.logger.info(f"[随机判断(回退)] 未触发绘画，概率: {paint_probability}")
                            return False
                except Exception as e:
                    self.logger.error(f"[智能判断] 使用AstrBot大模型接口时发生错误: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    # 请求失败，回退到随机判断模式
                    self.logger.info("[智能判断] 请求失败，回退到随机判断模式")
                    paint_probability = self.config.get("paint_probability", 0.3)
                    if random.random() < paint_probability:
                        self.logger.info(f"[随机判断(回退)] 随机触发绘画，概率: {paint_probability}")
                        return True
                    else:
                        self.logger.info(f"[随机判断(回退)] 未触发绘画，概率: {paint_probability}")
                        return False
                    
            except Exception as e:
                self.logger.error(f"[智能判断] 使用大模型进行判断时发生错误: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 发生错误时，回退到随机判断模式
                self.logger.info("[智能判断] 发生异常，回退到随机判断模式")
                paint_probability = self.config.get("paint_probability", 0.3)
                if random.random() < paint_probability:
                    self.logger.info(f"[随机判断(回退)] 随机触发绘画，概率: {paint_probability}")
                    return True
                else:
                    self.logger.info(f"[随机判断(回退)] 未触发绘画，概率: {paint_probability}")
                    return False
        
        # 默认不进行绘画
        self.logger.info("[智能判断] 未触发绘画")
        return False
    
    async def generate_prompt(self, message: str, event: AstrMessageEvent, llm_reply: str = "") -> str:
        """
        基于用户消息、大模型回复和对话历史以及人物预设生成相关绘画提示词
        
        Args:
            message: 用户消息
            event: 消息事件对象
            llm_reply: 大模型的回复内容
            
        Returns:
            生成的绘画提示词
        """
        try:
            # 获取用户ID和会话ID
            user_id = str(event.message_obj.sender.user_id)
            session_id = event.session_id
            
            # 获取系统消息历史
            self.logger.info(f"[提示词生成] 获取会话 {session_id} 的消息历史")
            
            # 使用正确的API方法获取消息历史
            # 从event中获取platform_id和user_id
            platform_id = event.platform_meta.id
            user_id = event.unified_msg_origin
            
            try:
                system_messages = await self.context.message_history_manager.get(
                    platform_id=platform_id,
                    user_id=user_id,
                    page=1,
                    page_size=5
                )
                
                conversation_text = ""
                if system_messages:
                    for msg in system_messages:
                        # 解析消息内容
                        if msg.content and isinstance(msg.content, dict):
                            content = msg.content.get('message_str', '')
                            sender_name = msg.sender_name or ''
                            role = "user" if sender_name != "AstrBot" else "assistant"
                            conversation_text += f"{role}: {content}\n"
                    self.logger.info(f"[提示词生成] 已获取 {len(system_messages)} 条系统消息历史")
                else:
                    self.logger.info("[提示词生成] 未获取到系统消息历史，使用插件内缓存")
                    # 使用插件内缓存作为备选
                    recent_conversations = [c for c in self.conversation_cache if c["user_id"] == user_id][-5:]
                    for conv in recent_conversations:
                        conversation_text += f"user: {conv['user_message']}\nassistant: {conv['bot_response']}\n"
            except Exception as e:
                self.logger.error(f"[提示词生成] 获取消息历史时发生错误: {str(e)}")
                self.logger.info("[提示词生成] 使用插件内缓存作为备选")
                # 使用插件内缓存作为备选
                recent_conversations = [c for c in self.conversation_cache if c["user_id"] == user_id][-5:]
                conversation_text = ""
                for conv in recent_conversations:
                    conversation_text += f"user: {conv['user_message']}\nassistant: {conv['bot_response']}\n"
            
            # 检测用户是否询问当前活动状态
            activity_query_phrases = ["在干嘛", "在做什么", "你在干什么", "你在做什么", "你在忙什么", "在忙什么"]
            is_activity_query = any(phrase in message for phrase in activity_query_phrases)
            
            # 构建提示词生成请求
            if is_activity_query:
                # 用户询问当前活动，使用活动相关的提示词
                activity_prompt = self.activity_images.get(self.current_activity, self.activity_images["none"])
                prompt_gen_prompt = f"""
Generate a detailed English prompt for an image based on the following activity and AI response:

Activity: {self.current_activity}

AI response: {llm_reply}

Please generate a detailed English prompt including:
1. Main subject and scene description (related to the current activity and AI response)
2. Art style and medium
3. Lighting and colors
4. Composition and perspective
5. Emotion and atmosphere appropriate for the activity and AI response

Return only the prompt, no additional explanation.
"""
            else:
                # 正常对话，使用上下文相关的提示词生成
                prompt_gen_prompt = f"""
Based on the following conversation history, current user message, and AI response, generate a detailed English prompt suitable for text-to-image models.

Conversation history:
{conversation_text}

Current user message: {message}

AI response: {llm_reply}

Important requirements:
- The generated image prompt must be closely connected to the conversation history, user message, and AI response, creating a coherent and natural continuation of events
- Avoid creating abrupt or unrelated scenes that don't fit with the ongoing conversation
- Ensure the image content logically follows from the dialogue context and AI response
- Maintain consistency with the established character profile and conversation tone

Please generate a detailed English prompt including:
1. Main subject and scene description (closely related to conversation context and AI response)
2. Art style and medium
3. Lighting and colors
4. Composition and perspective
5. Emotion and atmosphere that matches the conversation tone and AI response

Return only the prompt, no additional explanation.
"""
            
            self.logger.info(f"[提示词生成] 使用对话历史构建的提示词: {prompt_gen_prompt[:100]}...")
            
            # 使用系统自带的LLM生成更详细的绘画提示词
            self.logger.info("[提示词生成] 使用系统自带LLM生成绘画提示词")
            
            # 直接获取当前使用的LLM提供商实例
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
            if not provider:
                self.logger.error("[提示词生成] 未找到可用的LLM提供商")
                return message
            
            # 使用系统自带的LLM生成提示词
            llm_response = await self.context.llm_generate(
                chat_provider_id=provider.meta().id,
                prompt=prompt_gen_prompt,
                temperature=0.7,
                max_tokens=500,
                contexts=[]  # 确保contexts是一个空列表而不是None
            )
            
            if llm_response and hasattr(llm_response, 'completion_text') and llm_response.completion_text:
                generated_prompt = llm_response.completion_text.strip()
                self.logger.info(f"[提示词生成] 生成的绘画提示词: {generated_prompt}")
                return generated_prompt
            else:
                self.logger.error("[提示词生成] LLM生成失败")
                return message
            
        except Exception as e:
            self.logger.error(f"[提示词生成] 生成提示词时发生错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            return message
    

    
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent, *args, **kwargs) -> AsyncGenerator:
        """
        处理消息事件，用于智能绘画判断和对话
        
        Args:
            event: 消息事件对象
        
        Yields:
            消息结果
        """
        # 只处理文本消息
        if not event.message_str:
            return
        
        # 获取用户ID和消息
        user_id = str(event.message_obj.sender.user_id)
        message = event.message_str
        
        # 判断是否应该绘画
        should = await self.should_paint(message, user_id, event)
        
        if should:
            self.logger.info(f"[智能绘画] 触发绘画，用户: {user_id}, 消息: {message[:50]}...")
            
            # 1. 获取当前会话使用的聊天模型 ID
            umo = event.unified_msg_origin
            provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            
            # 2. 获取当前会话的对话历史
            conv_mgr = self.context.conversation_manager
            curr_cid = await conv_mgr.get_curr_conversation_id(umo)
            if curr_cid:
                conversation = await conv_mgr.get_conversation(umo, curr_cid)
                if conversation and conversation.history:
                    try:
                        # 将字符串格式的对话历史解析为列表
                        history_list = json.loads(conversation.history)
                        # 转换为 Message 对象列表
                        from astrbot.core.agent.message import Message
                        contexts = [Message(**msg) for msg in history_list]
                        self.logger.info(f"[智能绘画] 获取到对话历史，共 {len(contexts)} 条消息")
                    except json.JSONDecodeError:
                        contexts = None
                        self.logger.error("[智能绘画] 对话历史解析失败")
                    except Exception as e:
                        contexts = None
                        self.logger.error(f"[智能绘画] 转换对话历史为 Message 对象失败: {e}")
                else:
                    contexts = None
                    self.logger.info("[智能绘画] 未获取到对话历史")
            else:
                contexts = None
                self.logger.info("[智能绘画] 当前没有激活的对话")
            
            # 3. 调用大模型生成回复（使用对话历史）
            self.logger.info(f"[智能绘画] 调用大模型生成回复，provider_id: {provider_id}")
            llm_response = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=message,
                contexts=contexts,
                temperature=0.7,
                max_tokens=200
            )
            
            # 3. 发送回复给用户
            if llm_response and hasattr(llm_response, 'completion_text') and llm_response.completion_text:
                reply_text = llm_response.completion_text.strip()
                self.logger.info(f"[智能绘画] 大模型回复: {reply_text}")
                yield event.plain_result(reply_text)
            else:
                self.logger.error("[智能绘画] LLM回复生成失败")
                reply_text = ""
            
            # 4. 生成提示词（包含大模型的回复内容）
            prompt = await self.generate_prompt(message, event, llm_reply=reply_text)
            
            # 5. 生成图片
            image_url = await self.generate_image(prompt, user_id)
            
            if image_url:
                if image_url.startswith(('http://', 'https://')):
                    # 是URL，下载图片到本地
                    image_path = await self.download_image(image_url)
                else:
                    # 已经是本地路径
                    image_path = image_url
                
                if image_path:
                    # 发送图片
                    yield event.image_result(image_path)
                    self.logger.info(f"[智能绘画] 图片已发送: {image_path}")
                    # 添加到对话缓存
                    self.add_to_conversation_cache(user_id, message, f"已生成图片: {image_path}")
                else:
                    # 如果下载失败，发送URL
                    yield event.plain_result(f"图片生成成功: {image_url}")
                    self.logger.info(f"[智能绘画] 图片URL已发送: {image_url}")
                    # 添加到对话缓存
                    self.add_to_conversation_cache(user_id, message, f"已生成图片: {image_url}")
            else:
                error_msg = "抱歉，图片生成失败，请稍后再试。"
                yield event.plain_result(error_msg)
                self.logger.error("[智能绘画] 图片生成失败")
                # 添加到对话缓存
                self.add_to_conversation_cache(user_id, message, error_msg)
            
            # 更新最后绘画时间
            self.last_paint_time = time.time()
        else:
            # 如果不生成绘画，直接返回，让系统的默认流程处理消息
            self.logger.info(f"[对话] 不需要绘画，交给系统默认流程处理，用户: {user_id}, 消息: {message[:50]}...")
            return  # 直接返回，不做任何处理，让系统的默认对话模型处理
    
    @filter.command("aiimg")
    async def aiimg_command(self, event: AstrMessageEvent, prompt: str = "") -> AsyncGenerator:
        """
        命令生成图片
        
        Args:
            event: 消息事件对象
            prompt: 绘画提示词
        
        Yields:
            消息结果
        """
        """
        命令生成图片
        
        Args:
            event: 消息事件对象
            prompt: 绘画提示词
        """
        # 获取用户ID
        user_id = str(event.message_obj.sender.user_id)
        
        if not prompt:
            yield event.plain_result("请提供绘画提示词，例如：/aiimg 一只可爱的猫")
            return
        
        self.logger.info(f"[命令绘画] 用户: {user_id}, 提示词: {prompt[:50]}...")
        
        # 生成图片
        image_url = await self.generate_image(prompt, user_id)
        
        if image_url:
            if image_url.startswith(('http://', 'https://')):
                # 是URL，下载图片到本地
                image_path = await self.download_image(image_url)
            else:
                # 已经是本地路径
                image_path = image_url
            
            if image_path:
                # 发送图片
                yield event.image_result(image_path)
                self.logger.info(f"[命令绘画] 图片已发送: {image_path}")
                # 添加到对话缓存
                self.add_to_conversation_cache(user_id, f"/aiimg {prompt}", f"已生成图片: {image_path}")
            else:
                # 如果下载失败，发送URL
                yield event.plain_result(f"图片生成成功: {image_url}")
                self.logger.info(f"[命令绘画] 图片URL已发送: {image_url}")
                # 添加到对话缓存
                self.add_to_conversation_cache(user_id, f"/aiimg {prompt}", f"已生成图片: {image_url}")
        else:
            yield event.plain_result("抱歉，图片生成失败，请稍后再试。")
            self.logger.error("[命令绘画] 图片生成失败")