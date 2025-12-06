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
from astrbot.api.star import Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.platform import Platform
from astrbot.api import AstrBotConfig


@register("ms_aiimg", "AI绘画插件", "基于魔搭社区的AI绘画生成插件", "1.08")
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
        
        # 人物形象设定
        self.character_profile = ""
        
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
    
    def set_character_profile(self, profile: str):
        """设置人物形象"""
        self.character_profile = profile
        self.logger.info(f"[设定] 人物形象已设置: {profile[:50]}...")
    
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

    async def generate_image(self, prompt: str, user_id: str = None) -> Optional[str]:
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
        
        # 添加人物形象到提示词
        if self.character_profile:
            prompt = f"角色设定: {self.character_profile}\n{prompt}"
        
        # 构建请求参数
        url = f"{self.api_url}v1/stable_diffusion/text2img"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model or "AI-ModelScope/stable-diffusion-xl-base-1.0",
            "prompt": prompt,
            "negative_prompt": "low quality, worst quality, blurry, watermark, signature",
            "width": int(self.size.split("x")[0]) if "x" in self.size else 768,
            "height": int(self.size.split("x")[1]) if "x" in self.size else 512,
            "steps": 30,
            "guidance_scale": 7.5,
            "num_images": 1
        }
        
        try:
            self.logger.info(f"[图片生成] 开始生成图片，提示词: {prompt[:50]}...")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        images = result.get("images", [])
                        if images:
                            image_url = images[0].get("url")
                            if image_url:
                                self.logger.info(f"[图片生成] 图片生成成功，URL: {image_url}")
                                return image_url
                            else:
                                self.logger.error("[图片生成] 响应中未找到图片URL")
                        else:
                            self.logger.error("[图片生成] 响应中未找到图片")
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
    
    async def should_paint(self, message: str, user_id: str) -> bool:
        """
        判断是否应该触发绘画
        
        Args:
            message: 用户消息
            user_id: 用户ID
            
        Returns:
            是否应该绘画
        """
        # 如果未启用LLM智能判断，使用随机概率
        if not self.enable_llm_judge:
            import random
            should = random.random() < self.paint_probability
            self.logger.info(f"[智能判断] 随机判断结果: {should} (概率: {self.paint_probability})")
            return should
        
        # 检查是否在最小间隔时间内
        current_time = time.time()
        if current_time - self.last_paint_time < self.min_paint_interval:
            self.logger.info("[智能判断] 在最小间隔时间内，跳过绘画")
            return False
        
        # 如果没有配置LLM API，使用随机概率
        if not self.judge_llm_api_key or not self.judge_llm_api_url:
            import random
            should = random.random() < self.paint_probability
            self.logger.info(f"[智能判断] 未配置LLM API，使用随机判断结果: {should}")
            return should
        
        # 构建判断提示词
        judge_prompt = f"""
请判断以下用户消息是否适合生成图片：

用户消息: {message}

请考虑以下因素：
1. 消息是否描述了具体的场景、人物或物体
2. 消息是否包含视觉元素
3. 消息是否表达了情感或氛围
4. 消息是否过于抽象或难以可视化

请只回答"是"或"否"，不要添加其他解释。
"""
        
        try:
            # 构建请求参数
            headers = {
                "Authorization": f"Bearer {self.judge_llm_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.judge_llm_model,
                "messages": [
                    {"role": "user", "content": judge_prompt}
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.judge_llm_api_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                        should = "是" in content
                        self.logger.info(f"[智能判断] LLM判断结果: {content} -> {should}")
                        return should
                    else:
                        error_text = await response.text()
                        self.logger.error(f"[智能判断] API请求失败，状态码: {response.status}, 错误: {error_text}")
        except Exception as e:
            self.logger.error(f"[智能判断] 判断时发生错误: {str(e)}")
        
        # 出错时使用随机概率
        import random
        should = random.random() < self.paint_probability
        self.logger.info(f"[智能判断] 出错，使用随机判断结果: {should}")
        return should
    
    async def generate_prompt(self, message: str, user_id: str) -> str:
        """
        基于用户消息和对话历史生成绘画提示词
        
        Args:
            message: 用户消息
            user_id: 用户ID
            
        Returns:
            生成的绘画提示词
        """
        # 如果没有配置LLM API，直接使用用户消息
        if not self.prompt_llm_api_key or not self.prompt_llm_api_url:
            self.logger.info("[提示词生成] 未配置LLM API，直接使用用户消息")
            return message
        
        # 获取相关对话历史
        recent_conversations = [c for c in self.conversation_cache if c["user_id"] == user_id][-5:]
        conversation_text = ""
        for conv in recent_conversations:
            conversation_text += f"用户: {conv['user_message']}\n助手: {conv['bot_response']}\n"
        
        # 构建提示词生成请求
        prompt_gen_prompt = f"""
基于以下对话历史和当前用户消息，生成一个适合文生图模型的英文提示词。

人物形象设定: {self.character_profile}

对话历史:
{conversation_text}

当前用户消息: {message}

请生成一个详细的英文提示词，包括:
1. 主要主体和场景描述
2. 艺术风格和媒介
3. 光照和色彩
4. 构图和视角
5. 情感和氛围

只返回提示词，不要添加其他解释。
"""
        
        try:
            # 构建请求参数
            headers = {
                "Authorization": f"Bearer {self.prompt_llm_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.prompt_llm_model,
                "messages": [
                    {"role": "user", "content": prompt_gen_prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.prompt_llm_api_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                        self.logger.info(f"[提示词生成] 生成成功: {content[:100]}...")
                        return content
                    else:
                        error_text = await response.text()
                        self.logger.error(f"[提示词生成] API请求失败，状态码: {response.status}, 错误: {error_text}")
        except Exception as e:
            self.logger.error(f"[提示词生成] 生成提示词时发生错误: {str(e)}")
        
        # 出错时直接使用用户消息
        self.logger.info("[提示词生成] 出错，直接使用用户消息")
        return message
    
    async def handle_message(self, event: AstrMessageEvent) -> None:
        """
        处理消息事件，用于智能绘画判断
        
        Args:
            event: 消息事件对象
        """
        # 只处理文本消息
        if not event.message_str:
            return
        
        # 获取用户ID和消息
        user_id = str(event.sender.user_id)
        message = event.message_str
        
        # 判断是否应该绘画
        should = await self.should_paint(message, user_id)
        
        if should:
            self.logger.info(f"[智能绘画] 触发绘画，用户: {user_id}, 消息: {message[:50]}...")
            
            # 生成提示词
            prompt = await self.generate_prompt(message, user_id)
            
            # 生成图片
            image_url = await self.generate_image(prompt, user_id)
            
            if image_url:
                # 下载图片到本地
                image_path = await self.download_image(image_url)
                
                if image_path:
                    # 发送图片
                    chain = [image_path]
                    await self.send(event, chain)
                    self.logger.info(f"[智能绘画] 图片已发送: {image_path}")
                else:
                    # 如果下载失败，发送URL
                    await self.send(event, f"图片生成成功: {image_url}")
                    self.logger.info(f"[智能绘画] 图片URL已发送: {image_url}")
            else:
                await self.send(event, "抱歉，图片生成失败，请稍后再试。")
                self.logger.error("[智能绘画] 图片生成失败")
            
            # 更新最后绘画时间
            self.last_paint_time = time.time()
        
        # 添加到对话缓存
        # 注意：这里假设bot_response是空字符串，因为我们没有处理bot的实际回复
        # 在实际使用中，你可能需要从其他地方获取bot的回复
        self.add_to_conversation_cache(user_id, message, "")
    
    @filter.command("aiimg")
    async def aiimg_command(self, event: AstrMessageEvent, prompt: str = ""):
        """
        /aiimg 命令处理函数
        
        Args:
            event: 消息事件对象
            prompt: 图片生成提示词
        """
        # 获取用户ID
        user_id = str(event.sender.user_id)
        
        # 如果没有提供提示词，提示用户
        if not prompt:
            await self.send(event, "请提供图片生成提示词，例如：/aiimg 一只可爱的猫咪")
            return
        
        self.logger.info(f"[命令绘画] 用户: {user_id}, 提示词: {prompt}")
        
        # 生成图片
        image_url = await self.generate_image(prompt, user_id)
        
        if image_url:
            # 下载图片到本地
            image_path = await self.download_image(image_url)
            
            if image_path:
                # 发送图片
                chain = [image_path]
                await self.send(event, chain)
                self.logger.info(f"[命令绘画] 图片已发送: {image_path}")
            else:
                # 如果下载失败，发送URL
                await self.send(event, f"图片生成成功: {image_url}")
                self.logger.info(f"[命令绘画] 图片URL已发送: {image_url}")
        else:
            await self.send(event, "抱歉，图片生成失败，请稍后再试。")
            self.logger.error("[命令绘画] 图片生成失败")