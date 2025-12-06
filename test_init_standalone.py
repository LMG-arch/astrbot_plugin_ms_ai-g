#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试插件初始化 - 独立版本
不依赖astrbot模块，直接测试ModFlux类
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import importlib.util

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 创建简单的Mock类
class MockLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")
    
    def warning(self, msg):
        print(f"[WARNING] {msg}")
    
    def error(self, msg):
        print(f"[ERROR] {msg}")
    
    def debug(self, msg):
        print(f"[DEBUG] {msg}")

class MockContext:
    """模拟Context类"""
    def __init__(self):
        self.config = {}
        self.logger = MockLogger()
        self.get_config = self._get_config
        self.set_config = self._set_config
    
    def _get_config(self, key, default=None):
        return self.config.get(key, default)
    
    def _set_config(self, key, value):
        self.config[key] = value

class MockStar:
    """模拟Star基类"""
    def __init__(self, context=None):
        self.context = context

# 创建Mock对象
mock_logger = MockLogger()

# 创建模拟的astrbot模块
astrbot_mock = MagicMock()
astrbot_mock.api.all.logger = mock_logger
astrbot_mock.core.star.Star = MockStar

# 模拟register装饰器
registry = {}
def mock_register(name, description, detail, version):
    def decorator(cls):
        registry[name] = {
            'class': cls,
            'description': description,
            'detail': detail,
            'version': version
        }
        return cls
    return decorator

astrbot_mock.api.star.register = mock_register

# 将模拟的astrbot模块添加到sys.modules
sys.modules['astrbot'] = astrbot_mock
sys.modules['astrbot.api'] = astrbot_mock.api
sys.modules['astrbot.api.all'] = astrbot_mock.api.all
sys.modules['astrbot.api.star'] = astrbot_mock.api.star
sys.modules['astrbot.api.event'] = astrbot_mock.api.event
sys.modules['astrbot.api.platform'] = astrbot_mock.api.platform
sys.modules['astrbot.core'] = astrbot_mock.core
sys.modules['astrbot.core.star'] = astrbot_mock.core.star
sys.modules['astrbot.core.star.astr_bot_event'] = astrbot_mock.core.star.astr_bot_event
sys.modules['astrbot.core.star.context'] = astrbot_mock.core.star.context
sys.modules['astrbot.core.config'] = astrbot_mock.core.config
sys.modules['astrbot.core.config.astr_bot_config'] = astrbot_mock.core.config.astr_bot_config

# 加载main.py模块
spec = importlib.util.spec_from_file_location("main_module", os.path.join(current_dir, "main.py"))
main_module = importlib.util.module_from_spec(spec)

# 执行模块代码
spec.loader.exec_module(main_module)

# 获取ModFlux类
ModFlux = main_module.ModFlux

# 检查ModFlux类的类型
print(f"ModFlux类类型: {type(ModFlux)}")
print(f"ModFlux是否为Mock: {isinstance(ModFlux, Mock)}")

# 打印插件注册信息
print("=== 插件注册信息 ===")
print(f"插件元数据: {registry}")

# 测试1: 无参数初始化
print("\n=== 测试插件初始化 ===")
print("测试1: 无参数初始化")
try:
    plugin = ModFlux(None)
    print("✓ 测试1: 无参数初始化成功")
except Exception as e:
    print(f"✗ 测试1: 无参数初始化失败: {str(e)}")

# 测试2: 仅context参数初始化
print("\n测试2: 仅context参数初始化")
try:
    context = MockContext()
    plugin = ModFlux(context)
    print("✓ 测试2: 仅context参数初始化成功")
except Exception as e:
    print(f"✗ 测试2: 仅context参数初始化失败: {str(e)}")

# 测试3: context和config参数初始化
print("\n测试3: context和config参数初始化")
try:
    context = MockContext()
    config = {"api_key": "test_key", "model": "test_model"}
    plugin = ModFlux(context, config)
    print("✓ 测试3: context和config参数初始化成功")
except Exception as e:
    print(f"✗ 测试3: context和config参数初始化失败: {str(e)}")

# 测试4: 关键字参数初始化
print("\n测试4: 关键字参数初始化")
try:
    context = MockContext()
    config = {"api_key": "test_key", "model": "test_model"}
    plugin = ModFlux(context=context, config=config)
    print("✓ 测试4: 关键字参数初始化成功")
except Exception as e:
    print(f"✗ 测试4: 关键字参数初始化失败: {str(e)}")

# 测试5: 配置更新
print("\n测试5: 配置更新")
try:
    context = MockContext()
    plugin = ModFlux(context)
    new_config = {"api_key": "new_key", "model": "new_model"}
    plugin.on_config_update(new_config)
    print("✓ 测试5: 配置更新成功")
except Exception as e:
    print(f"✗ 测试5: 配置更新失败: {str(e)}")

# 测试6: 检查配置属性
print("\n测试6: 检查配置属性")
try:
    context = MockContext()
    config = {"api_key": "test_key", "model": "test_model", "size": "512x512"}
    plugin = ModFlux(context, config)
    
    # 检查配置是否正确设置
    print(f"API密钥: {plugin.api_key}")
    print(f"模型: {plugin.model}")
    print(f"尺寸: {plugin.size}")
    
    # 测试配置更新
    new_config = {"api_key": "new_key", "model": "new_model", "size": "1024x768"}
    plugin.on_config_update(new_config)
    
    print(f"更新后API密钥: {plugin.api_key}")
    print(f"更新后模型: {plugin.model}")
    print(f"更新后尺寸: {plugin.size}")
    
    print("✓ 测试6: 检查配置属性成功")
except Exception as e:
    print(f"✗ 测试6: 检查配置属性失败: {str(e)}")

# 测试7: 插件注册信息
print("\n测试7: 插件注册信息")
try:
    register_info = registry.get('ms_aiimg', {})
    print(f"插件注册信息: {register_info}")
    print("✓ 测试7: 插件注册信息获取成功")
except Exception as e:
    print(f"✗ 测试7: 插件注册信息获取失败: {str(e)}")