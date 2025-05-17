from llm_engine import LLMEngine
from typing import List, AsyncGenerator, Tuple, Dict, Union
import time

import pprint
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
        
        yield (f"{request_id}: ", False)
        yield (f"en...", False)
        for i in range(50):
            yield ("....", False)
            time.sleep(0.1)
            
        yield ("", True)