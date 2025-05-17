from llm_engine import LLMEngine
from typing import List, AsyncGenerator, Tuple, Dict, Union
import time

class MockLLMEngine(LLMEngine):
    def __init__(self):
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
        for i in range(50):
            yield (f"{i+1}-{i+1}-{i+1}-{i+1}", False)
            time.sleep(0.1)
            
        yield ("", True)