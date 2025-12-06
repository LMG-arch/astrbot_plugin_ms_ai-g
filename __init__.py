"""
插件元数据
"""

from astrbot.api.star import PluginMetadata
from .main import ModFlux

def create_plugin_metadata():
    """
    创建插件元数据
    
    Returns:
        PluginMetadata: 插件元数据对象
    """
    return PluginMetadata(
        name="ms_ai_g",
        version="1.08",
        description="接入魔搭社区文生图模型。支持LLM调用和命令调用。",
        author="LMG-arch",
        url="https://github.com/LMG-arch/astrbot_plugin_ms_ai-g.git",
        star_cls=ModFlux
    )