from items import (
    ConversationItem,
)
from abc import abstractmethod, ABC
import time
from typing import AsyncGenerator, List, Tuple, Dict, Union

class LLMEngine(ABC):
    @abstractmethod
    async def generate_response(
        self, 
        messages: list[Dict[str, Union[str, List[Dict[str, str]]]]], 
        temperature=0.2, 
        max_tokens=4096, 
        request_id=str(time.time())
    ) -> AsyncGenerator[Tuple[str, bool], None]:
        pass
    
    @abstractmethod
    def start(self):
        pass