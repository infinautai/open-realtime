#!/usr/bin/env python
# 文件名: test_tts_engine.py
# TTS 引擎测试脚本

import os
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
import time
from typing import List, Optional

from loguru import logger
from dotenv import load_dotenv

# 导入 TTS 引擎类
from tts_engine import TTSEngine
from tts.openai_tts import OpenAITTSService
from tts.elevenlabs_tts import ElevenLabsTTSService

# 加载环境变量
load_dotenv()

logger.remove()  # Remove existing handlers
logger.add(sys.stderr, level="DEBUG")


# 可用的 TTS 引擎类型
TTS_ENGINE_TYPES = {
    "openai": OpenAITTSService,
    "elevenlabs": ElevenLabsTTSService,
}

# TTS 引擎参数预设
TTS_PRESETS = {
    "openai": {
        "default": {
            "voice": "alloy",
            "model": "gpt-4o-mini-tts",
            "sample_rate": 16000,
        },
        "echo": {
            "voice": "echo",
            "model": "gpt-4o-mini-tts",
            "sample_rate": 16000,
        },
        "nova": {
            "voice": "nova",
            "model": "gpt-4o-mini-tts",
            "sample_rate": 16000,
        },
    },
    "elevenlabs": {
        "default": {
            "voice": "JBFqnCBsd6RMkjVDRZzb",  # 默认声音 ID
            "model": "eleven_multilingual_v2",
            "sample_rate": 16000,
        },
    }
}

# 测试用的文本样本
TEST_SAMPLES = [
    "这是一个简单的测试句子，用于测试 TTS 引擎的基本功能。",
    "Hello, this is a test sentence in English to check multilingual support.",
    "这是一个较长的句子，包含了更多的文本内容，用于测试 TTS 引擎在处理较长文本时的性能和质量。这种测试可以帮助我们了解模型在生成较长音频时的连贯性和自然度。",
    "问题：人工智能的未来发展趋势是什么？回答：人工智能未来将更加注重与人类的协作，强化学习和多模态模型将成为研究热点。"
]

# 首字延迟测试用的文本样本 (不同长度)
LATENCY_TEST_SAMPLES = [
    "测试。",  # 极短文本
    "这是一个短句子。",  # 短文本
    "这是一个稍长的句子，用于测试 TTS 引擎的首字延迟。",  # 中等长度文本
    "这是一个较长的句子，包含了更多的文本内容，用于测试 TTS 引擎在处理较长文本时的首字延迟表现。这种测试可以帮助我们了解模型处理不同长度文本的响应时间。",  # 长文本
]

class TTSEngineTester:
    """TTS 引擎测试器，用于测试不同的 TTS 引擎实现"""
    
    def __init__(self, output_dir: str = "./tts_tests"):
        """初始化测试器
        
        Args:
            output_dir: 输出音频文件的目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_engine: Optional[TTSEngine] = None
        self.engine_type: Optional[str] = None
    
    def _get_output_filename(self, text_id: int, engine_type: str, preset_name: str) -> Path:
        """生成输出文件名
        
        Args:
            text_id: 测试文本的 ID
            engine_type: 引擎类型
            preset_name: 预设名称
            
        Returns:
            输出文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"{engine_type}_{preset_name}_sample{text_id}_{timestamp}.wav"
    
    async def setup_engine(self, engine_type: str, preset_name: str = "default") -> None:
        """设置 TTS 引擎
        
        Args:
            engine_type: 引擎类型 ('openai' 或 'elevenlabs')
            preset_name: 预设名称
            
        Raises:
            ValueError: 如果引擎类型或预设名称无效
        """
        if engine_type not in TTS_ENGINE_TYPES:
            raise ValueError(f"未知的引擎类型: {engine_type}。支持的类型: {list(TTS_ENGINE_TYPES.keys())}")
        
        if preset_name not in TTS_PRESETS.get(engine_type, {}):
            raise ValueError(f"引擎 {engine_type} 没有预设 '{preset_name}'。可用预设: {list(TTS_PRESETS.get(engine_type, {}).keys())}")
        
        # 清理任何现有引擎
        if self.current_engine:
            self.current_engine.unload()
            self.current_engine = None
        
        # 获取引擎类和预设参数
        engine_class = TTS_ENGINE_TYPES[engine_type]
        preset_params = TTS_PRESETS[engine_type][preset_name].copy()
        
        # 获取 API 密钥环境变量
        if engine_type == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("未设置 OPENAI_API_KEY 环境变量")
            preset_params["api_key"] = api_key
        elif engine_type == "elevenlabs":
            api_key = os.environ.get("ELEVENLABS_API_KEY")
            if not api_key:
                raise ValueError("未设置 ELEVENLABS_API_KEY 环境变量")
            preset_params["api_key"] = api_key
        
        # 创建并加载引擎
        logger.info(f"正在初始化 {engine_type} TTS 引擎，预设: {preset_name}")
        self.current_engine = engine_class(**preset_params)
        self.current_engine.load()
        self.engine_type = engine_type
        
        logger.info(f"引擎设置完成: {engine_type} ({preset_name})")
    
    async def test_sample(self, text: str, text_id: int, preset_name: str = "default") -> Path:
        """测试单个文本样本
        
        Args:
            text: 要转换为语音的文本
            text_id: 测试文本的 ID
            preset_name: 预设名称
            
        Returns:
            保存的音频文件路径
        """
        if not self.current_engine:
            raise ValueError("引擎尚未初始化，请先调用 setup_engine")
        
        output_file = self._get_output_filename(text_id, self.engine_type, preset_name)
        
        logger.info(f"正在转换文本 {text_id} 为语音...")
        logger.debug(f"文本内容: '{text[:50]}...'")
        
        # 测量转换时间
        start_time = time.time()
        first_chunk_time = None
        chunk_index = 0
        
        # 创建文件并写入音频数据
        with open(output_file, "wb") as f:
            async for chunk in self.current_engine.run_tts(text):
                # 记录第一个块返回的时间
                if chunk_index == 0:
                    first_chunk_time = time.time()
                    first_chunk_latency = first_chunk_time - start_time
                    logger.info(f"首字延迟: {first_chunk_latency:.3f} 秒")
                
                f.write(chunk)
                # 打印进度
                print(".", end="", flush=True)
                chunk_index += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        print()  # 换行
        logger.info(f"转换完成，耗时: {duration:.2f} 秒")
        logger.info(f"音频保存至: {output_file}")
        
        return output_file
    
    async def test_custom_params(self, text: str, params: dict) -> Path:
        """使用自定义参数测试 TTS
        
        Args:
            text: 要转换为语音的文本
            params: 自定义 TTS 参数
            
        Returns:
            保存的音频文件路径
        """
        if not self.current_engine:
            raise ValueError("引擎尚未初始化，请先调用 setup_engine")
        
        # 应用自定义参数
        await self.current_engine.set_params(**params)
        
        # 自定义输出文件名
        param_str = "_".join(f"{k}_{v}" for k, v in params.items())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{self.engine_type}_custom_{param_str}_{timestamp}.wav"
        
        logger.info(f"使用自定义参数转换文本为语音...")
        logger.debug(f"文本内容: '{text[:50]}...'")
        logger.debug(f"参数: {params}")
        
        # 测量转换时间
        start_time = time.time()
        first_chunk_time = None
        chunk_index = 0
        
        # 创建文件并写入音频数据
        with open(output_file, "wb") as f:
            async for chunk in self.current_engine.run_tts(text):
                # 记录第一个块返回的时间
                if chunk_index == 0:
                    first_chunk_time = time.time()
                    first_chunk_latency = first_chunk_time - start_time
                    logger.info(f"首字延迟: {first_chunk_latency:.3f} 秒")
                
                f.write(chunk)
                # 打印进度
                print(".", end="", flush=True)
                chunk_index += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        print()  # 换行
        logger.info(f"转换完成，耗时: {duration:.2f} 秒")
        logger.info(f"音频保存至: {output_file}")
        
        return output_file
    
    async def run_full_test(self, engine_type: str, preset_name: str = "default") -> List[Path]:
        """运行完整测试，测试所有样本
        
        Args:
            engine_type: 引擎类型
            preset_name: 预设名称
            
        Returns:
            生成的音频文件路径列表
        """
        await self.setup_engine(engine_type, preset_name)
        
        output_files = []
        for i, sample in enumerate(TEST_SAMPLES):
            output_file = await self.test_sample(sample, i, preset_name)
            output_files.append(output_file)
        
        return output_files
    
    async def test_first_chunk_latency(self, engine_type: str, preset_name: str = "default") -> None:
        """测试首字延迟，测试不同长度文本的首字响应时间
        
        Args:
            engine_type: 引擎类型
            preset_name: 预设名称
        """
        await self.setup_engine(engine_type, preset_name)
        
        logger.info(f"开始测试 {engine_type} 引擎的首字延迟...")
        results = []
        
        for i, sample in enumerate(LATENCY_TEST_SAMPLES):
            logger.info(f"\n测试样本 {i+1}/{len(LATENCY_TEST_SAMPLES)} (长度: {len(sample)} 字符)")
            
            # 记录转换时间
            start_time = time.time()
            first_chunk_time = None
            chunk_index = 0
            
            # 只测量第一个音频块的延迟，不保存文件
            async for chunk in self.current_engine.run_tts(sample):
                if chunk_index == 0:
                    first_chunk_time = time.time()
                    first_chunk_latency = first_chunk_time - start_time
                    logger.info(f"首字延迟: {first_chunk_latency:.3f} 秒")
                    results.append((len(sample), first_chunk_latency))
                    break
                chunk_index += 1
        
        # 显示汇总结果
        logger.info("\n首字延迟测试结果汇总:")
        logger.info(f"{'文本长度':^10} | {'首字延迟(秒)':^12}")
        logger.info(f"{'-'*10:^10} | {'-'*12:^12}")
        for length, latency in results:
            logger.info(f"{length:^10} | {latency:.3f}")
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.current_engine:
            logger.info("正在卸载 TTS 引擎...")
            self.current_engine.unload()
            self.current_engine = None
        logger.info("清理完成")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TTS 引擎测试工具")
    parser.add_argument("--engine", type=str, choices=TTS_ENGINE_TYPES.keys(), default="openai",
                        help="要测试的 TTS 引擎类型")
    parser.add_argument("--preset", type=str, default="default",
                        help="TTS 引擎参数预设")
    parser.add_argument("--output-dir", type=str, default="./tts_tests",
                        help="音频输出目录")
    parser.add_argument("--text", type=str,
                        help="要转换的自定义文本")
    parser.add_argument("--list-presets", action="store_true",
                        help="列出所有可用的预设")
    parser.add_argument("--test-latency", action="store_true",
                        help="测试不同长度文本的首字延迟")
    
    args = parser.parse_args()
    
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # 列出预设
    if args.list_presets:
        for engine, presets in TTS_PRESETS.items():
            print(f"{engine} 引擎预设:")
            for preset_name, params in presets.items():
                print(f"  - {preset_name}: {params}")
        return
    
    # 创建测试器
    tester = TTSEngineTester(output_dir=args.output_dir)
    
    try:
        if args.test_latency:
            # 测试首字延迟
            await tester.test_first_chunk_latency(args.engine, args.preset)
        elif args.text:
            # 测试单个自定义文本
            await tester.setup_engine(args.engine, args.preset)
            await tester.test_sample(args.text, 0, args.preset)
        else:
            # 运行完整测试
            await tester.run_full_test(args.engine, args.preset)
    finally:
        # 清理资源
        tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
