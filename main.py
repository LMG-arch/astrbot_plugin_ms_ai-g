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

版本：1.0
"""

from astrbot.api.message_components import Plain, Image
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
import aiohttp
import asyncio
import random
import json
import base64
import os
import tempfile
import re
import time
from pathlib import Path


# 注册插件到AstrBot系统
@register("ms_aiimg", "", "接入魔搭社区文生图模型。支持LLM调用和命令调用。", "1.0")
class ModFlux(Star):
    """
    魔搭社区文生图插件主类
    
    继承自AstrBot的Star基类，提供图像生成功能
    """
    
    def __init__(self, context: Context, config: dict):
        """
        初始化插件
        
        Args:
            context: AstrBot上下文对象
            config: 插件配置字典
        """
        super().__init__(context)
        # 存储配置字典以便后续使用
        self.config = config
        
        # 从配置中获取API相关参数
        self.api_key = config.get("api_key")  # API密钥
        self.model = config.get("model")      # 模型名称
        self.size = config.get("size", "768x512")  # 默认图片尺寸
        self.api_url = config.get("api_url")  # API基础URL
        self.provider = config.get("provider", "ms")  # 提供商，默认为ModelScope
        
        # 智能绘画判断相关配置
        self.paint_probability = config.get("paint_probability", 0.3)  # 绘画触发概率，默认30%
        self.last_paint_time = 0  # 上次绘画时间
        self.min_paint_interval = config.get("min_paint_interval", 300)  # 最小绘画间隔，默认5分钟
        
        # LLM智能判断配置
        self.enable_llm_judge = config.get("enable_llm_judge", False)  # 是否启用LLM智能判断
        
        # 判断是否绘画的大模型配置
        self.judge_llm_api_url = config.get("judge_llm_api_url", "")  # 判断是否绘画的LLM API地址
        self.judge_llm_api_key = config.get("judge_llm_api_key", "")  # 判断是否绘画的LLM API密钥
        self.judge_llm_model = config.get("judge_llm_model", "gpt-3.5-turbo")  # 判断是否绘画的LLM模型名称
        
        # 生成提示词的大模型配置
        self.prompt_llm_api_url = config.get("prompt_llm_api_url", "")  # 生成提示词的LLM API地址
        self.prompt_llm_api_key = config.get("prompt_llm_api_key", "")  # 生成提示词的LLM API密钥
        self.prompt_llm_model = config.get("prompt_llm_model", "gpt-3.5-turbo")  # 生成提示词的LLM模型名称
        
        # 对话历史缓存（用于基于上下文的提示词生成）
        self.conversation_cache = []
        self.max_cache_size = config.get("max_cache_size", 10)  # 从配置读取最大缓存对话条数
        
        # 人物扮演形象配置
        self.character_profile = config.get("default_character_profile", "")  # 从配置读取默认人物扮演形象描述
        
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
        
        def set_character_profile(self, profile: str):
            """
            设置人物扮演形象描述
            
            Args:
                profile: 人物扮演形象描述文本
            """
            self.character_profile = profile
            print(f"已设置人物扮演形象：{profile}")
        
        # 创建临时目录用于存储下载的图片
        temp_dir_name = config.get("temp_dir_name", "astrbot_images")  # 从配置读取临时目录名称
        self.temp_dir = Path(tempfile.gettempdir()) / temp_dir_name
        self.temp_dir.mkdir(exist_ok=True)

        # 验证必要配置
        if not self.api_key:
            raise ValueError("API密钥必须配置")

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
                        
                        return str(file_path)
                    else:
                        raise Exception(f"下载图片失败，HTTP状态码: {response.status}")
        except Exception as e:
            raise Exception(f"图片下载失败: {str(e)}")

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
                        return base64.b64encode(image_data).decode('utf-8')
                    else:
                        raise Exception(f"获取图片数据失败，HTTP状态码: {response.status}")
        except Exception as e:
            raise Exception(f"图片转base64失败: {str(e)}")

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
        async with session.post(
            f"{self.api_url}v1/images/generations",
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
            async with session.get(
                f"{self.api_url}v1/tasks/{task_id}",
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

    @filter.llm_tool(name="draw")
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

            # 特殊的实例检查
            if self is None:
                yield event.plain_result("你还不够虔诚，所以没有得到佛祖的庇佑导致发生了错误")
                return
            
            # 方法1：下载图片到本地
            try:
                local_image_path = await self._download_image(image_url)
                chain = [Image.fromFile(local_image_path)]
            except Exception as download_error:
                # 方法1失败，尝试方法2：使用base64编码
                try:
                    image_base64 = await self._image_to_base64(image_url)
                    chain = [Image.fromBase64(image_base64)]
                except Exception as base64_error:
                    # 所有方法都失败，返回原始URL（可能会被平台拦截）
                    chain = [Image.fromURL(image_url)]
            
            yield event.chain_result(chain)

        except Exception as e:
            # 异常处理，返回错误信息
            yield event.plain_result(f"生成图片时遇到问题: {str(e)}")
            
    @filter.command("aiimg")
    async def generate_image_command(self, event: AstrMessageEvent):
        """
        命令调用接口 - 通过/aiimg命令生成图片
        
        用户可以通过发送"/aiimg <提示词>"命令来生成图片
        
        Args:
            event: AstrBot消息事件对象
            
        Yields:
            AstrBot消息事件结果，包含生成的图片和提示词信息或错误信息
        """
        # 解析用户输入的命令和提示词
        full_message = event.message_obj.message_str
        parts = full_message.split(" ", 1)  # 分割命令和提示词
        prompt = parts[1].strip() if len(parts) > 1 else ""

        # 验证提示词是否为空
        if not prompt:
            yield event.plain_result("\n请提供提示词！使用方法：/aiimg <提示词>")
            return

        try:
            # 生成随机种子
            current_seed = random.randint(1, 2147483647)

            # 调用图像生成API
            image_url = await self._request_image(prompt, self.size)
            
            # 尝试多种方式发送图片
            try:
                # 方法1：下载图片到本地
                local_image_path = await self._download_image(image_url)
                chain = [
                    Plain(f"提示词：{prompt}\n"),
                    Image.fromFile(local_image_path)
                ]
            except Exception as download_error:
                # 方法1失败，尝试方法2：使用base64编码
                try:
                    image_base64 = await self._image_to_base64(image_url)
                    chain = [
                        Plain(f"提示词：{prompt}\n"),
                        Image.fromBase64(image_base64)
                    ]
                except Exception as base64_error:
                    # 所有方法都失败，返回原始URL（可能会被平台拦截）
                    chain = [
                        Plain(f"提示词：{prompt}\n"),
                        Image.fromURL(image_url)
                    ]
            
            yield event.chain_result(chain)

        except Exception as e:
            # 异常处理，返回错误信息
            yield event.plain_result(f"\n生成图片失败: {str(e)}")

    @filter.command("setcharacter")
    async def set_character_command(self, event: AstrMessageEvent):
        """
        命令调用接口 - 通过/setcharacter命令设置人物扮演形象
        
        用户可以通过发送"/setcharacter <人物形象描述>"命令来设置人物扮演形象
        
        Args:
            event: AstrBot消息事件对象
            
        Yields:
            AstrBot消息事件结果，包含设置成功或失败信息
        """
        # 解析用户输入的命令和人物形象描述
        full_message = event.message_obj.message_str
        parts = full_message.split(" ", 1)  # 分割命令和人物形象描述
        profile = parts[1].strip() if len(parts) > 1 else ""

        # 验证人物形象描述是否为空
        if not profile:
            yield event.plain_result("\n请提供人物扮演形象描述！使用方法：/setcharacter <人物形象描述>")
            return

        try:
            # 设置人物扮演形象
            self.set_character_profile(profile)
            yield event.plain_result(f"\n✅ 人物扮演形象设置成功！\n\n当前形象：{profile}\n\n后续生成的绘画提示词将基于此人物形象和对话内容进行创作。")

        except Exception as e:
            # 异常处理，返回错误信息
            yield event.plain_result(f"\n设置人物扮演形象失败: {str(e)}")

    async def _should_paint(self, message: str) -> bool:
        """
        判断是否应该触发绘画功能
        
        Args:
            message: 用户消息内容
            
        Returns:
            bool: 是否触发绘画
        """
        # 检查时间间隔
        current_time = time.time()
        if current_time - self.last_paint_time < self.min_paint_interval:
            return False
        
        # 检查概率触发
        if random.random() > self.paint_probability:
            return False
        
        # 如果启用了LLM智能判断，优先使用LLM判断
        if self.enable_llm_judge and self.llm_api_url and self.llm_api_key:
            return await self._llm_judge_should_paint(message)
        
        # 否则使用传统的关键词匹配方法
        return await self._keyword_judge_should_paint(message)

    async def _llm_judge_should_paint(self, message: str) -> bool:
        """
        使用AI大语言模型判断是否应该触发绘画功能
        
        Args:
            message: 用户消息内容
            
        Returns:
            bool: 是否触发绘画
        """
        try:
            # 构建LLM判断请求
            prompt = f"""请分析以下用户消息内容，判断是否适合生成一幅画作。

用户消息："{message}"

请根据以下标准进行判断：
1. 消息是否包含绘画相关的关键词（如画、绘画、图片、图像、照片、插图、绘图等）
2. 消息是否包含描述性的内容（如风景、人物、动物、建筑、场景、故事等）
3. 消息是否具有视觉化的潜力（能够激发视觉想象）
4. 消息长度是否足够（至少10个字符）

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
                    {"role": "system", "content": "你是一个绘画触发判断助手，请严格根据标准判断是否应该生成画作。"},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 10,  # 固定最大令牌数
                "temperature": 0.1  # 固定温度参数
            }
            
            # 发送LLM请求
            async with aiohttp.ClientSession() as session:
                async with session.post(self.judge_llm_api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        llm_response = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
                        
                        # 解析LLM响应
                        if "是" in llm_response or "yes" in llm_response or "true" in llm_response:
                            return True
                        else:
                            return False
                    else:
                        # LLM请求失败，回退到关键词判断
                        print(f"LLM判断请求失败，状态码：{response.status}")
                        return await self._keyword_judge_should_paint(message)
                        
        except Exception as e:
            # LLM判断异常，回退到关键词判断
            print(f"LLM判断异常：{str(e)}")
            return await self._keyword_judge_should_paint(message)

    async def _keyword_judge_should_paint(self, message: str) -> bool:
        """
        使用传统关键词匹配方法判断是否应该触发绘画功能
        
        Args:
            message: 用户消息内容
            
        Returns:
            bool: 是否触发绘画
        """
        # 检查消息内容是否包含绘画相关关键词
        paint_keywords = [
            '画', '绘画', '图片', '图像', '照片', '插图', '绘图', '画画',
            '风景', '人物', '动物', '建筑', '场景', '故事', '想象',
            '美丽', '漂亮', '壮观', '梦幻', '奇幻', '科幻', '浪漫'
        ]  # 固定关键词列表
        
        # 检查消息长度，过短的消息不触发
        if len(message.strip()) < 10:  # 固定最小消息长度
            return False
        
        # 检查是否包含绘画关键词
        for keyword in paint_keywords:
            if keyword in message:
                return True
        
        # 检查是否包含描述性内容（形容词+名词的组合）
        descriptive_patterns = [
            r'的[^，。！？]{3,10}的',  # 包含"的...的"的描述性结构
            r'[美漂壮梦奇科浪][丽亮观幻幻技漫]',  # 形容词组合
            r'\w+的\w+',  # "A的B"结构
        ]
        
        for pattern in descriptive_patterns:
            if re.search(pattern, message):
                return True
        
        return False

    async def _generate_paint_prompt(self, message: str, conversation_history: list = None) -> str:
        """
        根据对话内容和对话历史生成绘画提示词
        
        Args:
            message: 用户当前消息内容
            conversation_history: 对话历史列表，包含之前的对话内容
            
        Returns:
            str: 生成的绘画提示词
        """
        # 如果启用了LLM智能判断且有生成提示词的大模型配置，优先使用AI模型生成提示词
        if self.enable_llm_judge and self.prompt_llm_api_url and self.prompt_llm_api_key:
            try:
                return await self._llm_generate_paint_prompt(message, conversation_history)
            except Exception as e:
                print(f"AI提示词生成失败，使用传统方法：{str(e)}")
        
        # 使用传统方法生成提示词
        return await self._traditional_generate_paint_prompt(message)

    async def _llm_generate_paint_prompt(self, message: str, conversation_history: list = None) -> str:
        """
        使用AI大语言模型生成更优质的绘画提示词，基于对话历史、当前内容和人物扮演形象
        
        Args:
            message: 用户当前消息内容
            conversation_history: 对话历史列表
            
        Returns:
            str: 生成的绘画提示词
        """
        # 构建对话上下文 - 基于消息接收者（人物扮演）的视角
        context = message
        if conversation_history:
            # 合并对话历史，保留最近的几条对话
            recent_count = 5  # 固定最近对话保留条数
            recent_history = conversation_history[-recent_count:]  # 保留最近5条对话
            history_text = "\n".join([f"{item.get('role', '用户')}: {item.get('content', '')}" for item in recent_history])
            context = f"对话历史：\n{history_text}\n\n当前消息：{message}"
        
        # 添加人物扮演形象信息 - 这是消息接收者的视角
        character_context = ""
        if self.character_profile:
            character_context = f"\n人物扮演形象（消息接收者的视角）：\n{self.character_profile}\n"
        
        # 构建LLM提示词生成请求 - 强调基于消息接收者视角
        prompt = f"""请根据以下对话上下文和人物扮演形象（消息接收者的视角），生成一个与当前对话情境高度契合的AI绘画提示词。

对话上下文：
{context}{character_context}

请从消息接收者（人物扮演角色）的视角分析对话的主题、情感基调、人物形象和具体情境，然后生成一个完全贴合对话内容和人物形象的绘画提示词：

1. **人物视角**：提示词必须基于消息接收者（人物扮演角色）的视角和感受
2. **形象契合**：提示词必须符合人物扮演形象的特征和风格
3. **主题契合**：提示词必须反映对话的核心主题和讨论内容
4. **情感一致**：体现对话的情感基调（如欢乐、浪漫、神秘、温馨等）
5. **情境还原**：准确还原对话中描述的场景、人物或事件
6. **细节丰富**：包含具体的视觉元素、色彩、光线、构图等细节
7. **风格匹配**：选择与对话氛围和人物形象相符的艺术风格
8. **语言要求**：使用英文描述，便于AI绘画模型理解
9. **长度控制**：100-200个字符之间

请确保生成的提示词完全基于对话内容和人物形象，特别是从消息接收者的视角出发，不要添加无关元素。

生成的提示词："""
        
        # 构建OpenAI格式的请求 - 使用生成提示词的大模型配置
        headers = {
            "Authorization": f"Bearer {self.prompt_llm_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.prompt_llm_model,
            "messages": [
                {"role": "system", "content": "你是一个专业的对话情境绘画提示词生成助手。你的任务是从消息接收者（人物扮演角色）的视角分析对话的完整上下文，包括主题、情感基调、具体情境，然后生成完全贴合对话内容和人物形象的AI绘画提示词。提示词必须准确反映对话的核心内容，特别是从消息接收者的视角出发，不要添加任何与对话无关的元素。"},
            {"role": "user", "content": prompt}
            ],
            "max_tokens": 300,  # 固定最大令牌数
            "temperature": 0.7  # 固定温度参数
        }
        
        # 发送LLM请求
        async with aiohttp.ClientSession() as session:
            async with session.post(self.prompt_llm_api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    llm_response = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    
                    # 清理响应内容，确保只返回提示词
                    llm_response = re.sub(r'^[\"\']|[\"\']$', '', llm_response)  # 移除引号
                    llm_response = re.sub(r'^提示词：|^生成的提示词：', '', llm_response, flags=re.IGNORECASE)
                    
                    # 验证提示词质量
                    if len(llm_response) >= 20 and len(llm_response) <= 300:  # 固定长度限制
                        return llm_response
                    else:
                        raise Exception(f"生成的提示词长度不符合要求，应在20-300字符之间")
                else:
                    raise Exception(f"LLM提示词生成请求失败，状态码：{response.status}")
                        
    async def _traditional_generate_paint_prompt(self, message: str) -> str:
        """
        使用传统方法生成绘画提示词
        
        Args:
            message: 用户消息内容
            
        Returns:
            str: 生成的绘画提示词
        """
        # 基础提示词模板
        base_prompts = [
            "高清摄影，{content}，精美细节，专业构图，4K画质",
            "数字艺术，{content}，梦幻风格，光影效果，大师级作品",
            "插画风格，{content}，动漫风格，色彩鲜艳，艺术感强",
            "油画风格，{content}，古典艺术，质感丰富，艺术气息",
            "水彩画，{content}，柔和色彩，梦幻效果，艺术创作"
        ]
        
        # 提取消息中的关键内容
        content = message.strip()
        
        # 移除命令和特殊符号
        content = re.sub(r'^/\w+\s*', '', content)  # 移除命令前缀
        content = re.sub(r'[，。！？；：、]', ' ', content)  # 替换标点符号
        
        # 选择随机的基础提示词模板
        base_prompt = random.choice(base_prompts)
        
        # 生成最终的绘画提示词
        prompt = base_prompt.format(content=content)
        
        # 限制提示词长度
        if len(prompt) > 200:  # 固定最大长度
            prompt = prompt[:200] + "..."
        
        return prompt

    @filter.message()
    async def auto_paint_check(self, event: AstrMessageEvent):
        """
        自动绘画检查 - 在聊天过程中智能判断是否应该生成图片
        
        Args:
            event: AstrBot消息事件对象
            
        Yields:
            如果触发绘画，则返回生成的图片；否则不返回任何内容
        """
        # 获取用户消息
        message = event.message_obj.message_str
        
        # 跳过命令消息（避免重复触发）
        if message.startswith('/'):
            return
        
        # 更新对话历史缓存
        self._update_conversation_cache(message, "用户")
        
        # 判断是否应该触发绘画
        if await self._should_paint(message):
            try:
                # 更新最后绘画时间
                self.last_paint_time = time.time()
                
                # 生成绘画提示词（基于对话上下文）
                paint_prompt = await self._generate_paint_prompt(message, self.conversation_cache)
                
                # 调用图像生成API
                image_url = await self._request_image(paint_prompt, self.size)
                
                # 尝试多种方式发送图片
                try:
                    # 方法1：下载图片到本地
                    local_image_path = await self._download_image(image_url)
                    chain = [
                        Plain(f"根据我们的对话，我为你创作了一幅画：\n{paint_prompt}\n"),
                        Image.fromFile(local_image_path)
                    ]
                except Exception as download_error:
                    # 方法1失败，尝试方法2：使用base64编码
                    try:
                        image_base64 = await self._image_to_base64(image_url)
                        chain = [
                            Plain(f"根据我们的对话，我为你创作了一幅画：\n{paint_prompt}\n"),
                            Image.fromBase64(image_base64)
                        ]
                    except Exception as base64_error:
                        # 所有方法都失败，返回原始URL（可能会被平台拦截）
                        chain = [
                            Plain(f"根据我们的对话，我为你创作了一幅画：\n{paint_prompt}\n"),
                            Image.fromURL(image_url)
                        ]
                
                # 将绘画结果添加到对话历史缓存
                bot_response = f"根据我们的对话，我为你创作了一幅画：{paint_prompt}"
                self._update_conversation_cache(bot_response, "机器人")
                
                yield event.chain_result(chain)
                
            except Exception as e:
                # 绘画失败时静默处理，不干扰正常对话
                print(f"自动绘画失败: {str(e)}")
                
                # 将失败信息也添加到对话历史缓存
                error_response = f"绘画失败：{str(e)}"
                self._update_conversation_cache(error_response, "机器人")