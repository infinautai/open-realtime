from typing import AsyncGenerator, Dict, Literal, Optional

from loguru import logger
from audio.utils import create_default_resampler
from tts_engine import TTSEngine
from elevenlabs.client import ElevenLabs
import os


class ElevenLabsTTSService(TTSEngine):
    """
    """

    SERVICE_SAMPLE_RATE = 16000
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        voice: str = "JBFqnCBsd6RMkjVDRZzb",
        model: str = "eleven_multilingual_v2",
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        # Validate voice parameter immediately
     
        self._voice_id = voice
        self.model_name = model
        api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self._client = ElevenLabs(api_key=api_key)
        self._resampler = create_default_resampler()
        self.sample_rate = sample_rate or self.SERVICE_SAMPLE_RATE

    async def set_params(self, **kwargs) -> None:
        """Update TTS engine parameters.

        Args:
            **kwargs: Parameters to update, which may include:
                voice (str): Voice ID (alloy, echo, fable, onyx, nova, shimmer)
                model (str): Model name
                instructions (str): Voice customization instructions
                sample_rate (int): Audio sample rate in Hz
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if 'voice' in kwargs:
            voice = kwargs['voice']
           
            self._voice_id = voice
            logger.info(f"Switched TTS voice to: [{voice}]")

        if 'model' in kwargs:
            model = kwargs['model']
            self.model_name = model
            logger.info(f"Switched TTS model to: [{model}]")


    def load(self) -> None:
        """Load the TTS service. No-op for ElevenLabs TTS."""
        logger.info(f"Starting ElevenLabs TTS service with model: {self.model_name}")

    def unload(self) -> None:
        """Unload the TTS service. No-op for ElevenLabs TTS."""
        logger.info(f"Stopping ElevenLabs TTS service with model: {self.model_name}")


    async def run_tts(self, text: str, **kwargs) -> AsyncGenerator[bytes, None]:
        """
        """
        logger.debug(f"Generating TTS for text: [{text}]")
        try:

            audio_stream = self._client.text_to_speech.convert_as_stream(
                text=text,
                voice_id=self._voice_id,
                model_id=self.model_name,
                output_format="pcm_16000"
            )

            for chunk in audio_stream:
                if isinstance(chunk, bytes):
                    # Resample the audio if necessary
                    if self.sample_rate != self.SERVICE_SAMPLE_RATE:
                        # Resample the audio chunk to the desired sample rate
                        chunk = await self._resampler.resample(chunk,
                                                                self.SERVICE_SAMPLE_RATE,
                                                                self.sample_rate)
                    # Yield the audio chunk
                    yield chunk

        except Exception as e:
            logger.exception(f"{self} error generating TTS: {e}")
