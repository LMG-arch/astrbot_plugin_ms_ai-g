"""
插件元数据
"""

from .main import ModFlux

# 插件注册信息
plugin_metadata = {
    "ms_aiimg": {
        "class": ModFlux,
        "description": "AI绘画插件",
        "detail": "基于魔搭社区的AI绘画生成插件", 
        "version": "1.08"
    }
}