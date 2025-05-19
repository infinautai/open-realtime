#!/usr/bin/env python3
# Test the enhanced endofsentence detection

from utils.string import match_endofsentence

def test_endofsentence():
    test_cases = [
        # English sentences
        ("Hello, world! This is a test.", 13),  # Should detect at !
        ("Hello, world. This is a test.", 13),  # Should detect at .
        ("Hello, Mr. Smith. How are you?", 19),  # Should not detect at Mr.
        
        # Chinese sentences
        ("你好，世界。这是一个测试。", 6),  # Should detect at 。
        ("今天天气真好！你去哪里？", 7),  # Should detect at ！
        ("他说：\"我明天会来。\"然后就走了。", 10),  # Should detect at 。inside quotes
        ("问题是：如何解决这个难题？", 13),  # Should detect at ？
        ("这件衣服（红色的）很好看。", 12),  # Should detect at 。
        
        # Mixed language
        ("Hello，你好。This is a test.", 10),  # Should detect at 。
        ("Email test: user.name@example.com. Next sentence.", 34),  # Should not break at email
        ("数字测试：3.14159。这是下一句。", 12),  # Should not break at number
        
        # Special cases
        ("这段话结束了……下一段话开始了。", 8),  # Should detect at ……
        ("他说\"是的\"，然后离开了。", 9),  # Should detect at after quotes
    ]
    
    for idx, (text, expected_pos) in enumerate(test_cases):
        result = match_endofsentence(text)
        print(f"Test {idx+1}: {text}")
        print(f"  Expected: {expected_pos}, Got: {result}")
        print(f"  {'PASS' if result == expected_pos else 'FAIL: ' + text[:result] + '|' + text[result:]}")
        print()

if __name__ == "__main__":
    test_endofsentence()
