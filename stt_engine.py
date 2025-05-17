from abc import abstractmethod, ABC
from typing import AsyncGenerator

class STTEngine(ABC):
    @abstractmethod
    async def run_stt(self, audio: bytes) -> AsyncGenerator[str, None]:
        """Transcribe given audio using STT engine."""
        pass
    @abstractmethod
    def load(self):
        """Load STT engine model."""
        pass
    @abstractmethod
    def unload(self):
        """Unload STT engine model."""
        pass
