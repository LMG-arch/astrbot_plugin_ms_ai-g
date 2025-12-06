"""
魔搭社区文生图插件 for AstrBot

该插件提供基于魔搭社区（ModelScope）的AI图像生成功能，支持LLM工具调用和命令调用两种方式。

功能特性：
- 支持魔搭社区文生图API
- 支持异步图像生成任务轮询
- 提供LLM工具调用接口（draw方法）
- 提供命令调用接口（/aiimg命令）
- 支持自定义图片尺寸和模型参数
- 自动处理QQ平台URL限制问题（下载图片到本地）
- 智能绘画判断功能，聊天过程中自动识别绘画机会
- 支持AI大语言模型智能判断，符合OpenAI格式

版本：1.07
"""

from astrbot.api.message_components import Plain, Image
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import llm_tool
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger
import aiohttp
import asyncio
import random
import json
import base64
import tempfile
import re
import time
import hashlib
from pathlib import Path


# 注册插件到AstrBot系统
@register(name="ms_ai-g", desc="接入魔搭社区文生图模型。支持LLM调用和命令调用。", version="1.07", author="LMG-arch")
class ModFlux(Star):
    """
    魔搭社区文生图插件主类
    
    继承自AstrBot的Star基类，提供图像生成功能
    """
    
    def __init__(self, context: Context, config: AstrBotConfig = None):
        """
        初始化插件
        
        Args:
            context: AstrBot上下文对象
            config: 插件配置对象
        """
        super().__init__(context)
        
        # 使用AstrBot提供的logger接口
        self.logger = logger
        
        # 初始化配置变量（这些将通过on_config_update方法设置）
        self.config = config if config is not None else {}
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
                        import hashlib
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

    async def _request_modelscope(self, prompt: str, size: str, session: aiohttp.ClientSession) -> str:
        """
        向魔搭社区API发送图像生成请求
        
        Args:
            prompt: 图像生成提示词
            size: 图像尺寸（如"1920x1080"）
            session: aiohttp会话对象
            
        Returns:
            str: 生成的图像URL
            
        Raises:
            Exception: 当API请求失败或图像生成失败时抛出
        """
        # 设置通用请求头
        common_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 生成随机种子以确保每次生成不同的图像
        current_seed = random.randint(1, 2147483647)
        
        # 构建请求负载
        payload = {
            "model": f"{self.model}",
            "prompt": prompt,
            "seed": current_seed,
            "size": size,
            "num_inference_steps": "30",  # 固定推理步数
        }
        
        # 发送异步图像生成请求
        # 修复URL构建问题，避免重复的v1路径
        if self.api_url and self.api_url.rstrip('/').endswith('/v1'):
            api_endpoint = f"{self.api_url.rstrip('/')}/images/generations"
        else:
            api_endpoint = f"{self.api_url.rstrip('/')}/v1/images/generations"
        async with session.post(
            api_endpoint,
            headers={**common_headers, "X-ModelScope-Async-Mode": "true"},  # 启用异步模式
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8')
        ) as response:
            response.raise_for_status()
            task_response = await response.json()
            task_id = task_response.get("task_id")
            
            if not task_id:
                raise Exception("未能获取任务ID，生成图片失败。")

        # 使用指数退避策略轮询任务结果
        delay = 1  # 固定初始延迟
        max_delay = 10  # 固定最大延迟
        
        while True:
            # 查询任务状态
            # 修复URL构建问题，避免重复的v1路径
            if self.api_url and self.api_url.rstrip('/').endswith('/v1'):
                task_endpoint = f"{self.api_url.rstrip('/')}/tasks/{task_id}"
            else:
                task_endpoint = f"{self.api_url.rstrip('/')}/v1/tasks/{task_id}"
            async with session.get(
                task_endpoint,
                headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
            ) as result_response:
                result_response.raise_for_status()
                data = await result_response.json()

                task_status = data.get("task_status")
                
                if task_status == "SUCCEED":
                    # 任务成功，返回生成的图像URL
                    output_images = data.get("output_images", [])
                    if output_images:
                        return output_images[0]
                    else:
                        raise Exception("图片生成成功但未返回图片URL。")
                elif task_status == "FAILED":
                    # 任务失败
                    raise Exception("图片生成失败。")
                
                # 任务仍在进行中，使用指数退避策略等待
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)  # 指数增长延迟时间，但不超过最大值

    async def _request_image(self, prompt: str, size: str) -> str:
        """
        统一的图像生成请求入口
        
        根据配置的提供商调用相应的API方法
        
        Args:
            prompt: 图像生成提示词
            size: 图像尺寸
            
        Returns:
            str: 生成的图像URL
            
        Raises:
            ValueError: 当提示词为空或不支持的提供商时抛出
            Exception: 网络请求或解析失败时抛出
        """
        try:
            # 验证API密钥是否配置
            if not self.api_key:
                raise ValueError("API密钥未配置，请前往插件配置页面设置API密钥。")
            
            # 验证提示词
            if not prompt:
                raise ValueError("请提供提示词！")

            # 根据提供商选择相应的API调用方法
            async with aiohttp.ClientSession() as session:
                if self.provider.lower() == "ms" or self.provider.lower() == "modelscope":
                    return await self._request_modelscope(prompt, size, session)
                else:
                    raise ValueError(f"不支持的提供商: {self.provider}")

        except aiohttp.ClientError as e:
            # 网络请求异常
            raise Exception(f"网络请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            # JSON解析异常
            raise Exception(f"解析API响应失败: {str(e)}")
        except Exception as e:
            # 其他异常直接抛出
            raise e

    @llm_tool(name="draw")
    async def draw(self, event: AstrMessageEvent, prompt: str, size: str = "768x512"):
        '''
        LLM工具调用接口 - 根据提示词生成图片
        
        该方法可以被AstrBot的LLM系统调用，用于智能对话中的图像生成
        
        Args:
            prompt(string): 图片提示词，需要包含主体、场景、风格等必要提示词
            size(string): 图片尺寸，如1920x1080，默认768x512
            
        Yields:
            AstrBot消息事件结果，包含生成的图片或错误信息
        '''
        
        try:
            # 调用图像生成API
            image_url = await self._request_image(prompt, size)

            # 方法1：下载图片到本地
            try:
                local_image_path = await self._download_image(image_url)
                # 直接使用event.send发送图片，避免使用chain_result
                await event.send([Image.fromFileSystem(local_image_path)])
                self.logger.info("[工具调用] 图片发送完成")
                return
                
            except Exception:
                # 方法1失败，尝试方法2：使用base64编码
                try:
                    image_base64 = await self._image_to_base64(image_url)
                    # 直接使用event.send发送图片，避免使用chain_result
                    await event.send([Image.fromURL(f"data:image/png;base64,{image_base64}")])
                    self.logger.info("[工具调用] 图片发送完成")
                    return
                    
                except Exception:
                    # 所有方法都失败，返回原始URL（可能会被平台拦截）
                    # 直接使用event.send发送URL，避免使用chain_result
                    await event.send([Image.fromURL(image_url)])
                    self.logger.info("[工具调用] 图片发送完成")
                    return

        except Exception as e:
             # 异常处理，静默处理错误，不发送任何文字信息
             self.logger.error(f"生成图片时遇到问题: {str(e)}")
             return

    async def _should_paint(self, message: str, conversation_history: list = None) -> bool:
        """
        根据用户要求，仅使用AI大模型判断是否应该触发绘画功能，并在判断后应用概率控制
        
        Args:
            message: 用户当前消息内容
            conversation_history: 对话历史列表，包含之前的对话内容
            
        Returns:
            bool: 是否应该绘画
        """
        self.logger.debug(f"[AI绘图判断] 开始使用AI大模型判断是否需要绘图，消息内容: {message}")
        self.logger.info(f"[绘图概率] 当前绘图触发概率: {self.paint_probability}")
        
        # 检查是否配置了判断用的AI大模型
        if not self.judge_llm_api_url or not self.judge_llm_api_key:
            self.logger.warning("[AI绘图判断] 未配置判断用的AI大模型API地址或密钥，无法进行AI判断")
            return False
        
        # 强制使用AI大模型判断是否绘画（基于完整对话上下文）
        self.logger.info("[AI绘图判断] 将对话和历史记录发送给AI大模型进行判断...")
        result, llm_response = await self._llm_judge_should_paint(message, conversation_history)
        self.logger.debug(f"[AI绘图判断] LLM判断结果: {result}")
        
        # 在LLM判断结果基础上应用概率控制
        if result:
            import random
            random_number = random.random()
            self.logger.info(f"[概率控制] 生成随机数: {random_number}, 阈值: {self.paint_probability}")
            should_paint = random_number < self.paint_probability
        else:
            should_paint = False
            self.logger.info(f"[概率控制] LLM判断为否，不进行概率检查")
        
        # 在终端显示最终判断结果
        self.logger.info(f"[最终判断] 是否需要绘画: {'是' if should_paint else '否'}")
        return should_paint



    async def _llm_judge_should_paint(self, message: str, conversation_history: list = None) -> tuple:
        """
        使用AI大语言模型判断是否应该触发绘画功能，基于当前对话和对话历史
        
        Args:
            message: 用户当前消息内容
            conversation_history: 对话历史列表，包含之前的对话内容
            
        Returns:
            tuple: (是否触发绘画, LLM回复内容)
        """
        try:
            self.logger.debug(f"[LLM绘图判断] 开始使用LLM判断是否需要绘图，消息内容: {message}")
            
            # 构建对话上下文
            context = ""
            if conversation_history and len(conversation_history) > 0:
                # 取最近的对话记录
                recent_history = conversation_history[-3:]  # 最近3条对话
                context = "\n对话历史：\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
                self.logger.debug(f"[LLM绘图判断] 对话历史: {recent_history}")
            
            # 构建LLM判断请求 - 基于完整的对话上下文
            prompt = f"""请分析以下对话内容，判断当前对话是否适合生成一幅画作。

当前对话：
用户最新消息："{message}"
{context}

请根据以下标准进行判断：
1. 对话是否包含绘画相关的主题或关键词（如画、绘画、图片、图像、照片、插图、绘图等）
2. 对话是否包含描述性的内容（如风景、人物、动物、建筑、场景、故事等视觉化元素）
3. 对话是否具有视觉化的潜力（能够激发视觉想象，形成具体的画面）
4. 对话是否具有情感共鸣或故事性（能够通过绘画增强表达效果）
5. 对话上下文是否连贯，有足够的背景信息支持绘画创作

请综合考虑整个对话的上下文，而不是只看最新消息。如果对话适合生成画作来增强表达或回应，请回答"是"，否则回答"否"。

请只回答"是"或"否"，不需要解释原因。

判断结果："""
            
            # 构建OpenAI格式的请求
            headers = {
                "Authorization": f"Bearer {self.judge_llm_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.judge_llm_model,
                "messages": [
                    {"role": "system", "content": "你是一个绘画触发判断助手，请基于完整的对话上下文判断是否应该生成画作来增强对话表达。"},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 10,  # 固定最大令牌数
                "temperature": 0.1  # 固定温度参数
            }
            
            self.logger.debug("[LLM绘图判断] 发送LLM判断请求...")
            
            # 发送LLM请求
            async with aiohttp.ClientSession() as session:
                async with session.post(self.judge_llm_api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        llm_response = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
                        self.logger.debug(f"[LLM绘图判断] LLM响应内容: {llm_response}")
                        
                        # 在终端显示AI大语言模型的判断结果
                        self.logger.info(f"[AI判断结果] 大模型判断回复: {llm_response}")
                        
                        # 解析LLM响应
                        if "是" in llm_response or "yes" in llm_response or "true" in llm_response:
                            self.logger.debug("[LLM绘图判断] LLM判断结果: 是")
                            return True, llm_response
                        else:
                            self.logger.debug("[LLM绘图判断] LLM判断结果: 否")
                            return False, llm_response
                    else:
                        # LLM请求失败，直接抛出异常，不回退到其他判断方法
                        self.logger.error(f"[LLM绘图判断] LLM请求失败，状态码：{response.status}")
                        raise Exception(f"LLM请求失败，状态码：{response.status}")
                        
        except Exception as e:
            # LLM判断异常，直接抛出异常，不回退到其他判断方法
            self.logger.error(f"[LLM绘图判断] LLM判断异常：{str(e)}")
            raise Exception(f"LLM判断异常：{str(e)}")

    async def _keyword_judge_should_paint(self, message: str) -> bool:
        """
        使用传统关键词匹配方法判断是否应该触发绘画功能
        
        Args:
            message: 用户消息内容
            
        Returns:
            bool: 是否触发绘画
        """
        self.logger.debug(f"[关键词绘图判断] 开始使用关键词匹配判断是否需要绘图，消息内容: {message}")
        
        # 检查消息内容是否包含绘画相关关键词
        paint_keywords = [
            '画', '绘画', '图片', '图像', '照片', '插图', '绘图', '画画',
            '风景', '人物', '动物', '建筑', '场景', '故事', '想象',
            '美丽', '漂亮', '壮观', '梦幻', '奇幻', '科幻', '浪漫'
        ]  # 固定关键词列表
        
        # 检查消息长度，过短的消息不触发
        if len(message.strip()) < 10:  # 固定最小消息长度
            self.logger.debug(f"[关键词绘图判断] 消息长度不足，长度: {len(message.strip())}")
            return False
        self.logger.debug("[关键词绘图判断] 消息长度满足条件")
        
        # 检查是否包含绘画关键词
        for keyword in paint_keywords:
            if keyword in message:
                self.logger.debug(f"[关键词绘图判断] 匹配到关键词: {keyword}")
                return True
        self.logger.debug("[关键词绘图判断] 未匹配到任何关键词")
        
        # 检查是否包含描述性内容（形容词+名词的组合）
        descriptive_patterns = [
            r'的[^，。！？]{3,10}的',  # 包含"的...的"的描述性结构
            r'[美漂壮梦奇科浪][丽亮观幻幻技漫]',  # 形容词组合
            r'\w+的\w+',  # "A的B"结构
        ]
        
        for pattern in descriptive_patterns:
            if re.search(pattern, message):
                self.logger.debug(f"[关键词绘图判断] 匹配到描述性模式: {pattern}")
                return True
        self.logger.debug("[关键词绘图判断] 未匹配到任何描述性模式")
        
        self.logger.debug("[关键词绘图判断] 关键词匹配结果: 否")
        return False

    async def _generate_paint_prompt(self, message: str, conversation_history: list = None) -> str:
        """
        根据用户要求，仅使用AI大模型生成绘画指令
        
        Args:
            message: 用户当前消息内容
            conversation_history: 对话历史列表，包含之前的对话内容
            
        Returns:
            str: 生成的绘画提示词
        """
        self.logger.debug(f"[AI提示词生成] 开始使用AI大模型生成绘画指令，消息内容: {message}")
        
        # 检查是否配置了生成提示词的AI大模型
        if not self.prompt_llm_api_url or not self.prompt_llm_api_key:
            self.logger.error("[AI提示词生成] 未配置生成提示词的AI大模型API地址或密钥，无法生成绘画指令")
            raise Exception("未配置生成提示词的AI大模型API地址或密钥")
        
        # 强制使用AI大模型生成绘画指令（基于对话历史和人物形象）
        self.logger.info("[AI提示词生成] 将对话和历史记录发送给AI大模型生成绘画指令...")
        result = await self._llm_generate_paint_prompt(message, conversation_history)
        self.logger.debug(f"[AI提示词生成] LLM生成绘画指令完成: {result}")
        return result

    async def _llm_generate_paint_prompt(self, message: str, conversation_history: list = None) -> str:
        """
        使用AI大语言模型生成更优质的绘画提示词，基于对话历史、当前内容和人物扮演形象
        
        Args:
            message: 用户当前消息内容
            conversation_history: 对话历史列表
            
        Returns:
            str: 生成的绘画提示词
        """
        # 构建对话上下文
        context = ""
        if conversation_history and len(conversation_history) > 0:
            # 取最近的对话记录
            recent_history = conversation_history[-5:]  # 最近5条对话
            context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
            self.logger.info(f"[LLM提示词生成] 对话历史: {context}")
        
        # 添加人物扮演信息
        character_context = ""
        if hasattr(self, 'character_profile') and self.character_profile:
            character_context = f"\n人物扮演形象（消息接收者的视角）：\n{self.character_profile}\n"
            self.logger.info(f"[LLM提示词生成] 人物扮演形象: {self.character_profile}")
        
        # 构建LLM提示词生成请求 - 严格按照流程：基于对话历史、用户配置的人物形象生成绘画指令
        prompt = f"""请严格按照以下流程生成AI绘画提示词：

**输入信息：**
- 当前对话：{message}
- 对话历史：{context}
- 人物形象：{character_context}

**生成流程要求：**
1. **分析对话历史**：理解整个对话的上下文和主题
2. **结合人物形象**：基于消息接收者（人物扮演角色）的视角和特征
3. **生成绘画指令**：创建完全贴合对话内容和人物形象的提示词

**提示词要求：**
- 必须基于对话历史和人物形象，不能跳跃或并行处理
- 必须从消息接收者的视角出发
- 必须反映对话的核心主题和情感基调
- 必须包含具体的视觉元素和细节
- 使用英文描述，100-200个字符

生成的提示词："""
        
        # 构建OpenAI格式的请求 - 使用生成提示词的大模型配置
        headers = {
            "Authorization": f"Bearer {self.prompt_llm_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.prompt_llm_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        self.logger.info(f"[LLM提示词生成] 发送请求到LLM服务: 模型={self.prompt_llm_model}")
        
        try:
            # 发送请求到LLM服务
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.prompt_llm_api_url,  # 修复API地址引用错误
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        # 提取生成的提示词
                        generated_prompt = result["choices"][0]["message"]["content"].strip()
                        self.logger.info(f"[LLM提示词生成] 成功生成提示词: {generated_prompt}")
                        # 在终端显示AI大语言模型生成的提示词
                        self.logger.info(f"[AI提示词生成] 大模型生成的绘画提示词: {generated_prompt}")
                        return generated_prompt
                    else:
                        self.logger.error(f"[LLM提示词生成] LLM服务返回错误: 状态码={response.status}, 响应={await response.text()}")
                        # LLM调用失败，回退到传统方法
                        self.logger.warning("[LLM提示词生成] LLM调用失败，回退到传统方法生成提示词")
                        return await self._traditional_generate_paint_prompt(message)
        except Exception as e:
            self.logger.error(f"[LLM提示词生成] 调用LLM服务时发生异常: {str(e)}")
            # 发生异常时，回退到传统方法
            self.logger.warning("[LLM提示词生成] 发生异常，回退到传统方法生成提示词")
            return await self._traditional_generate_paint_prompt(message)
    
    def _extract_keywords(self, text: str) -> list:
        """
        从文本中提取关键词（简单实现）
        
        Args:
            text: 输入文本
            
        Returns:
            list: 提取出的关键词列表
        """
        # 简单的关键词提取实现 - 移除常见停用词并提取有意义的词汇
        # 在实际应用中，可以使用jieba分词或其他NLP库进行更精确的关键词提取
        
        # 常见停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        # 简单分词和清洗
        words = re.findall(r'[\\w]+', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        return keywords

    async def _traditional_generate_paint_prompt(self, message: str) -> str:
        """
        使用传统方法（模板+关键词）生成绘画提示词
        
        Args:
            message: 用户当前消息内容
            
        Returns:
            str: 生成的绘画提示词
        """
        self.logger.info(f"[传统提示词生成] 开始使用传统方法生成提示词，消息内容: {message}")
        
        # 使用基础模板
        base_templates = [
            "Anime style, {scene_description}, beautiful detailed eyes, soft lighting",
            "High quality anime art, {scene_description}, detailed facial features, vibrant colors",
            "Chibi style illustration, {scene_description}, cute character design, pastel colors"
        ]
        
        # 随机选择一个模板
        template = random.choice(base_templates)
        self.logger.info(f"[传统提示词生成] 选择的模板: {template}")
        
        # 提取关键词作为场景描述
        keywords = self._extract_keywords(message)
        self.logger.info(f"[传统提示词生成] 提取到的关键词: {keywords}")
        
        if keywords:
            scene_desc = ", ".join(keywords[:3])  # 最多使用3个关键词
        else:
            # 如果没有提取到关键词，则使用整个消息作为场景描述
            scene_desc = message[:50]  # 限制长度
            self.logger.info(f"[传统提示词生成] 未提取到关键词，使用消息前50字符作为场景描述: {scene_desc}")
        
        # 生成最终提示词
        prompt = template.format(scene_description=scene_desc)
        self.logger.info(f"[传统提示词生成] 传统方法生成提示词完成: {prompt}")
        return prompt

    async def auto_paint_check(self, event: AstrMessageEvent):
        """
        自动绘画检查 - 严格按照流程执行：
        1. AI根据对话和历史记录判断是否绘画
        2. 是的话：将对话、历史记录及用户配置的人物形象发送给大模型生成绘画指令
        3. 将绘画指令传递给绘画大模型生成图片
        4. 不绘画时正常聊天
        
        Args:
            event: AstrMessageEvent对象，包含消息的完整信息
        
        Yields:
            生成的图片或空结果
        """
        self.logger.info("[自动绘图] 开始自动绘图检查，严格按照流程执行")
        
        # 获取消息内容
        message = event.message_str
        
        if not message:
            # 无法获取消息内容，记录错误并返回
            self.logger.error(f"[自动绘图] 无法获取消息内容，事件对象类型: {type(event)}")
            return
            
        self.logger.info(f"[自动绘图] 接收到消息: {message}")
        
        # 跳过命令消息（避免重复触发）
        if message.startswith('/'):
            self.logger.info("[自动绘图] 消息以'/'开头，跳过绘图检查")
            return
        
        # 更新对话历史缓存
        self._update_conversation_cache(message, "用户")
        self.logger.info("[自动绘图] 已更新对话历史缓存")
        
        # 步骤1: AI根据对话和历史记录判断是否绘画
        self.logger.info("[自动绘图] 步骤1: AI根据对话和历史记录判断是否绘画...")
        should_paint = await self._should_paint(message, self.conversation_cache)
        
        if should_paint:
            self.logger.info("[自动绘图] 判断结果: 需要绘图，检查时间间隔...")
            
            # 检查是否满足最小绘图间隔要求
            current_time = time.time()
            time_since_last_paint = current_time - self.last_paint_time
            self.logger.info(f"[绘图间隔检查] 距离上次绘图已过: {time_since_last_paint:.1f}秒，最小间隔要求: {self.min_paint_interval}秒")
            
            if time_since_last_paint < self.min_paint_interval:
                remaining_time = self.min_paint_interval - time_since_last_paint
                self.logger.info(f"[绘图间隔检查] 距离上次绘图时间不足，还需等待 {remaining_time:.1f} 秒，跳过本次绘图")
                return
            
            self.logger.info("[自动绘图] 间隔检查通过，开始执行绘画流程")
            try:
                # 更新最后绘画时间
                self.last_paint_time = time.time()
                self.logger.info(f"[自动绘图] 已更新最后绘画时间: {self.last_paint_time}")
                
                # 步骤2: 将对话、历史记录及用户配置的人物形象发送给大模型生成绘画指令
                self.logger.info("[自动绘图] 步骤2: 将对话、历史记录及用户配置的人物形象发送给大模型生成绘画指令...")
                paint_prompt = await self._generate_paint_prompt(message, self.conversation_cache)
                self.logger.info(f"[自动绘图] 生成的绘画指令: {paint_prompt}")
                # 在终端显示最终生成的绘画提示词
                self.logger.info(f"[最终提示词] 绘画提示词: {paint_prompt}")
                
                # 步骤3: 将绘画指令传递给绘画大模型生成图片
                self.logger.info(f"[自动绘图] 步骤3: 将绘画指令传递给绘画大模型生成图片，提示词: {paint_prompt}，尺寸: {self.size}")
                image_url = await self._request_image(paint_prompt, self.size)
                self.logger.info(f"[自动绘图] 图片生成完成，图片URL: {image_url}")
                
                # 尝试多种方式发送图片（只发送图片，不添加文字描述）
                self.logger.info("[自动绘图] 开始发送图片...")
                try:
                    # 方法1：下载图片到本地
                    self.logger.info("[自动绘图] 尝试下载图片到本地...")
                    local_image_path = await self._download_image(image_url)
                    self.logger.info(f"[自动绘图] 图片下载完成，本地路径: {local_image_path}")
                    
                    # 直接使用event.send发送图片，避免使用chain_result
                    await event.send([Image.fromFileSystem(local_image_path)])
                    self.logger.info("[自动绘图] 图片发送完成")
                    return
                    
                except Exception as download_error:
                    self.logger.error(f"[自动绘图] 下载图片失败: {str(download_error)}")
                    # 方法1失败，尝试方法2：使用base64编码
                    try:
                        self.logger.info("[自动绘图] 尝试使用base64编码发送图片...")
                        image_base64 = await self._image_to_base64(image_url)
                        self.logger.info("[自动绘图] 图片base64编码完成")
                        
                        # 直接使用event.send发送图片，避免使用chain_result
                        await event.send([Image.fromURL(f"data:image/png;base64,{image_base64}")])
                        self.logger.info("[自动绘图] 图片发送完成")
                        return
                        
                    except Exception as base64_error:
                        self.logger.error(f"[自动绘图] base64编码失败: {str(base64_error)}")
                        # 所有方法都失败，返回原始URL（可能会被平台拦截）
                        self.logger.warning("[自动绘图] 所有方法都失败，返回原始URL")
                        
                        # 直接使用event.send发送URL，避免使用chain_result
                        await event.send([Image.fromURL(image_url)])
                        self.logger.info("[自动绘图] 图片发送完成")
                        return
                
            except Exception as e:
                # 绘画失败时静默处理，不干扰正常对话
                self.logger.error(f"[自动绘图] 自动绘画失败: {str(e)}")
                
                # 将失败信息也添加到对话历史缓存
                error_response = f"绘画失败：{str(e)}"
                self._update_conversation_cache(error_response, "机器人")
                self.logger.info("[自动绘图] 已将失败信息添加到对话历史缓存")
        else:
            # 步骤4: 不绘画时正常聊天
            self.logger.info("[自动绘图] 判断结果: 不需要绘图，正常聊天")
    
    def on_config_update(self, new_config: AstrBotConfig):
        """
        配置更新回调 - 当插件配置被更新时调用
        
        Args:
            new_config: 更新后的配置对象（AstrBotConfig）
        """
        self.logger.info("[配置更新] 接收到新的配置")
        
        # 处理AstrBotConfig对象，确保能正确提取配置值
        config_dict = {}
        if isinstance(new_config, dict):
            config_dict = new_config
        elif hasattr(new_config, '__dict__'):
            # 如果是对象，转换为字典
            config_dict = vars(new_config)
        elif hasattr(new_config, 'get'):
            # 如果已经有get方法，直接使用
            config_dict = new_config
        else:
            self.logger.warning("[配置更新] 无法识别的配置类型，使用空配置")
        
        # 更新配置字典
        self.config = config_dict if isinstance(config_dict, dict) else {}
        
        # 更新API相关参数
        self.api_key = self.config.get("api_key", self.api_key)
        self.model = self.config.get("model", self.model)
        self.size = self.config.get("size", self.size)
        self.api_url = self.config.get("api_url", self.api_url)
        self.provider = self.config.get("provider", self.provider)
        
        # 更新智能绘画判断相关配置
        self.paint_probability = self.config.get("paint_probability", self.paint_probability)
        self.min_paint_interval = self.config.get("min_paint_interval", self.min_paint_interval)
        
        # 更新LLM智能判断配置
        self.enable_llm_judge = self.config.get("enable_llm_judge", self.enable_llm_judge)
        
        # 更新判断是否绘画的大模型配置
        self.judge_llm_api_url = self.config.get("judge_llm_api_url", self.judge_llm_api_url)
        self.judge_llm_api_key = self.config.get("judge_llm_api_key", self.judge_llm_api_key)
        self.judge_llm_model = self.config.get("judge_llm_model", self.judge_llm_model)
        
        # 更新生成提示词的大模型配置
        self.prompt_llm_api_url = self.config.get("prompt_llm_api_url", self.prompt_llm_api_url)
        self.prompt_llm_api_key = self.config.get("prompt_llm_api_key", self.prompt_llm_api_key)
        self.prompt_llm_model = self.config.get("prompt_llm_model", self.prompt_llm_model)
        
        # 更新对话历史缓存配置
        self.max_cache_size = self.config.get("max_cache_size", self.max_cache_size)
        
        # 更新人物扮演形象配置
        self.character_profile = self.config.get("default_character_profile", self.character_profile)
        
        # 更新临时目录配置
        temp_dir_name = self.config.get("temp_dir_name", self.temp_dir_name)
        if temp_dir_name != self.temp_dir_name:
            self.temp_dir_name = temp_dir_name
            self.temp_dir = Path(tempfile.gettempdir()) / self.temp_dir_name
            self.temp_dir.mkdir(exist_ok=True)
        
        # 验证必要配置
        if not self.api_key:
            self.logger.warning("API密钥未配置，部分功能将受限。请前往插件配置页面设置API密钥。")
        else:
            self.logger.info("API密钥已配置，插件功能正常")
        
        self.logger.info("[配置更新] 配置已成功更新")
    
    @filter.command("aiimg")
    async def aiimg_command(self, event: AstrMessageEvent, prompt: str):
        """
        处理/aiimg命令 - 手动触发图像生成，严格按照流程执行：
        1. 接收用户提供的提示词
        2. 将提示词传递给绘画大模型生成图片
        3. 发送生成的图片
        
        Args:
            event: AstrMessageEvent对象，包含消息的完整信息
            prompt: 绘画提示词
            
        Yields:
            生成的图片或错误信息
        """
        self.logger.info(f"[命令处理] 接收到/aiimg命令，提示词: {prompt}")
        
        try:
            # 步骤2: 将提示词传递给绘画大模型生成图片
            self.logger.info(f"[命令处理] 步骤2: 将提示词传递给绘画大模型生成图片，提示词: {prompt}，尺寸: {self.size}")
            image_url = await self._request_image(prompt, self.size)
            self.logger.info(f"[命令处理] 图片生成完成，图片URL: {image_url}")
            
            # 步骤3: 发送生成的图片
            self.logger.info("[命令处理] 步骤3: 开始发送图片...")
            try:
                # 方法1：下载图片到本地
                self.logger.info("[命令处理] 尝试下载图片到本地...")
                local_image_path = await self._download_image(image_url)
                self.logger.info(f"[命令处理] 图片下载完成，本地路径: {local_image_path}")
                
                # 直接使用event.send发送图片，避免使用chain_result
                await event.send([Image.fromFileSystem(local_image_path)])
                self.logger.info("[命令处理] 图片发送完成")
                return
                
            except Exception as download_error:
                self.logger.error(f"[命令处理] 下载图片失败: {str(download_error)}")
                # 方法1失败，尝试方法2：使用base64编码
                try:
                    self.logger.info("[命令处理] 尝试使用base64编码发送图片...")
                    image_base64 = await self._image_to_base64(image_url)
                    self.logger.info("[命令处理] 图片base64编码完成")
                    
                    # 直接使用event.send发送图片，避免使用chain_result
                    await event.send([Image.fromURL(f"data:image/png;base64,{image_base64}")])
                    self.logger.info("[命令处理] 图片发送完成")
                    return
                    
                except Exception as base64_error:
                    self.logger.error(f"[命令处理] base64编码失败: {str(base64_error)}")
                    # 所有方法都失败，返回原始URL（可能会被平台拦截）
                    self.logger.warning("[命令处理] 所有方法都失败，返回原始URL")
                    
                    # 直接使用event.send发送URL，避免使用chain_result
                    await event.send([Image.fromURL(image_url)])
                    self.logger.info("[命令处理] 图片发送完成")
                    return
            
        except Exception as e:
            # 异常处理，静默处理错误，不发送任何文字信息
            error_msg = f"生成图片失败: {str(e)}"
            self.logger.error(f"[命令处理] {error_msg}")
            return

    async def on_message(self, event: AstrMessageEvent):
        """
        消息事件处理器 - 处理所有接收到的消息
        
        Args:
            event: AstrMessageEvent对象，包含消息的完整信息
        """
        # 调用自动绘画检查功能
        await self.auto_paint_check(event)