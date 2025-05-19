#!/usr/bin/env python3

import sys
import os

# 确保能正确导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.string import match_endofsentence

def test_first_sentence_break():
    """测试找出第一个句子断点的功能"""
    test_cases = [
        'Hello world!',               # English with !
        'Hello, Mr. Smith.',          # English with . after Mr.
        '今天天气很好。明天呢？',      # Chinese with 。 and ？
        '你好！我是AI。',              # Chinese with ！ and 。
        'Email: user.name@example.com. Next sentence.',  # With email
        '数字 3.14159 是圆周率。',     # With number
        'Prof. Smith is a doctor.',   # With abbreviation
        '他说："我明天来。"然后就走了。'  # With quotes
    ]
    
    for t in test_cases:
        pos = match_endofsentence(t)
        if pos > 0:
            print(f'"{t}" -> "{t[:pos]}"|"{t[pos:]}" (pos={pos})')
        else:
            print(f'"{t}" -> 没有找到句子断点 (pos={pos})')

def test_incremental_text():
    """测试渐进式文本的句子断点识别（模拟实时输入）"""
    texts = [
        # 英文示例
        "Hello! How are you? I am fine.",
        # 中文示例
        "你好！最近怎么样？我很好。今天天气真不错。"
    ]
    
    for full_text in texts:
        print(f"\n渐进式测试: '{full_text}'")
        buffer = ""
        
        for i in range(len(full_text)):
            # 增加一个字符
            buffer += full_text[i]
            print(f"  缓冲区: '{buffer}'")
            
            # 检查是否有句子可以提交
            pos = match_endofsentence(buffer)
            if pos > 0:
                print(f"  ✓ 发现句子结束: '{buffer[:pos]}'")
                # 保留剩余部分继续处理
                buffer = buffer[pos:]
                print(f"  → 剩余缓冲区: '{buffer}'")

        # 处理最后可能剩余的文本
        if buffer:
            print(f"  ➜ 最终剩余文本: '{buffer}'")

if __name__ == "__main__":
    print("=== 基本句子断点检测 ===")
    test_first_sentence_break()
    
    print("\n=== 渐进式文本处理测试 ===")
    test_incremental_text()
