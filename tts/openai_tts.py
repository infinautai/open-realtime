from typing import AsyncGenerator, Dict, Literal, Optional

from loguru import logger
from openai import AsyncOpenAI, BadRequestError
from audio.utils import create_default_resampler
from tts_engine import TTSEngine
import asyncio


ValidVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

VALID_VOICES: Dict[str, ValidVoice] = {
    "alloy": "alloy",
    "echo": "echo",
    "fable": "fable",
    "onyx": "onyx",
    "nova": "nova",
    "shimmer": "shimmer",
}


class OpenAITTSService(TTSEngine):
    """OpenAI Text-to-Speech service that generates audio from text.

    This service uses the OpenAI TTS API to generate PCM-encoded audio at 24kHz.

    Args:
        api_key: OpenAI API key. Defaults to None.
        base_url: Optional base URL for the OpenAI API. Defaults to None.
        voice: Voice ID to use. Must be one of: alloy, echo, fable, onyx, nova, shimmer. Defaults to "alloy".
        model: TTS model to use. Defaults to "gpt-4o-mini-tts".
        sample_rate: Output audio sample rate in Hz. If different from 24kHz, audio will be resampled. Defaults to None.
        instructions: Optional instructions to customize the voice. Defaults to None.
        **kwargs: Additional keyword arguments.

    The service returns PCM-encoded audio at the specified sample rate.
    """

    OPENAI_SAMPLE_RATE = 24000  # OpenAI TTS always outputs at 24kHz

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        voice: str = "alloy",
        model: str = "gpt-4o-mini-tts",
        sample_rate: Optional[int] = None,
        instructions: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        # Validate voice parameter immediately
        if voice not in VALID_VOICES:
            raise ValueError(f"Invalid voice '{voice}'. Must be one of {list(VALID_VOICES.keys())}")
        
        # Set initial state with validated parameters
        self.sample_rate = sample_rate or self.OPENAI_SAMPLE_RATE
        self.model_name = model
        self._voice_id = voice
        self._instructions = instructions
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._resampler = create_default_resampler()

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
            if voice not in VALID_VOICES:
                raise ValueError(f"Invalid voice '{voice}'. Must be one of {list(VALID_VOICES.keys())}")
            self._voice_id = voice
            logger.info(f"Switched TTS voice to: [{voice}]")

        if 'model' in kwargs:
            model = kwargs['model']
            self.model_name = model
            logger.info(f"Switched TTS model to: [{model}]")

        if 'instructions' in kwargs:
            self._instructions = kwargs['instructions']
            logger.info(f"Updated TTS instructions")

        if 'sample_rate' in kwargs:
            sample_rate = kwargs['sample_rate']
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                raise ValueError(f"Invalid sample rate: {sample_rate}")
            self.sample_rate = sample_rate
            logger.info(f"Updated sample rate to: {sample_rate} Hz")

    def load(self) -> None:
        """Load the TTS service. No-op for OpenAI TTS."""
        logger.info(f"Starting OpenAI TTS service with model: {self.model_name}")

    def unload(self) -> None:
        """Unload the TTS service. No-op for OpenAI TTS."""
        logger.info(f"Stopping OpenAI TTS service with model: {self.model_name}")


    async def run_tts(self, text: str, **kwargs) -> AsyncGenerator[bytes, None]:
        """Generate audio from text using OpenAI's TTS API.

        Args:
            text: The text to convert to speech.
            **kwargs: Additional parameters (ignored for OpenAI TTS).

        Yields:
            bytes: Audio data chunks in PCM format.

        Raises:
            BadRequestError: If the OpenAI API request fails.
        """
        logger.debug(f"Generating TTS for text: [{text}]")
        try:
            # Setup extra body parameters
            extra_body = {}
            if self._instructions:
                extra_body["instructions"] = self._instructions

            async with self._client.audio.speech.with_streaming_response.create(
                input=text or " ",  # Text must contain at least one character
                model=self.model_name,
                voice=VALID_VOICES[self._voice_id],
                response_format="pcm",
                extra_body=extra_body,
            ) as r:
                if r.status_code != 200:
                    error = await r.text()
                    logger.error(f"Error getting audio (status: {r.status_code}, error: {error})")
                    return


                CHUNK_SIZE = 1024

                async for chunk in r.iter_bytes(CHUNK_SIZE):
                    if len(chunk) > 0:
                        resampled_audio = chunk
                        if self.sample_rate != self.OPENAI_SAMPLE_RATE:
                            # Resample the audio to the desired sample rate
                            resampled_audio = await self._resampler.resample(
                                bytes(chunk), self.OPENAI_SAMPLE_RATE, self.sample_rate
                            )

                        yield resampled_audio

        except BadRequestError as e:
            logger.exception(f"{self} error generating TTS: {e}")
