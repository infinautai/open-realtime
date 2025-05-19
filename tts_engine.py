from abc import abstractmethod, ABC
from typing import AsyncGenerator, Any, Dict

class TTSEngine(ABC):
    @abstractmethod
    async def run_tts(self, text: str, **kwargs) -> AsyncGenerator[bytes, None]:
        """Generate TTS audio from text using TTS engine.
        
        Args:
            text: The text to convert to speech
            **kwargs: Additional parameters for the TTS engine (e.g. voice, speed, etc.)
        
        Returns:
            An async generator that yields audio chunks as bytes
        """
        pass

    @abstractmethod
    async def set_params(self, **kwargs) -> None:
        """Update TTS engine parameters.
        
        Args:
            **kwargs: Parameters to update, which may include:
                - voice: Voice ID or name
                - model: Model name or ID
                - sample_rate: Audio sample rate in Hz
                - instructions: Voice customization instructions
                - Other engine-specific parameters
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """Load TTS engine model."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload TTS engine model."""
        pass
