from llm_engine import LLMEngine
from typing import List, AsyncGenerator, Tuple, Dict, Union
import time

import pprint


test_response1 = "从前有一位名叫李青的侠客，他行侠仗义，心怀正义。"
test_response2 = """
一天，李青得知一个邪恶的黑帮在江湖上作恶多端，他们横行乡里，欺压百姓。李青决定揭露黑帮的罪行，保护无辜百姓。
他深入敌营，智斗黑帮的头目。在一场惊心动魄的对决中，李青运用了高超的武艺和智慧，终于将黑帮头目绳之以法。江湖因此恢复了平静，百姓们感激不尽，李青也因为他的英勇和正义，成为了后人传颂的侠义典范
"""

class MockLLMEngine(LLMEngine):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Omni-3B"):
        super().__init__()

    def start(self):
        # Initialize the engine
        self.initialized = True
        print("QwenOmniLLMEngine started")

    async def generate_response(
        self,
        messages: list[Dict[str, Union[str, List[Dict[str, str]]]]],
        temperature=0.2,
        max_tokens=4096,
        request_id=str(time.time()),
    ) -> AsyncGenerator[Tuple[str, bool], None]:
        # Add any additional initialization logic here
        
        pprint.pprint(messages)
        
        # yield (f"{request_id}: ", False)
        # yield (f"en...", False)
        # for i in range(50):
        #     yield ("....", False)
        #     time.sleep(0.1)
            
        # yield ("", True)
        
        yield (test_response1, False)
        # response 2 words in a time
        for i in range(0, len(test_response2), 2):
            yield (test_response2[i:i+2], False)
            time.sleep(0.1)
        yield ("", True)