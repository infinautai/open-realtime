import asyncio
from enum import Enum
from typing import AsyncGenerator, Optional, Union
import numpy as np
from loguru import logger
from stt.language import Language
from stt_engine import STTEngine

def language_to_whisper_language(language: Language) -> Optional[str]:
    """Maps pipecat Language enum to Whisper language codes.

    Args:
        language: A Language enum value representing the input language.

    Returns:
        str or None: The corresponding Whisper language code, or None if not supported.

    Note:
        Only includes languages officially supported by Whisper.
    """
    language_map = {
        # Arabic
        Language.AR: "ar",
        Language.AR_AE: "ar",
        Language.AR_BH: "ar",
        Language.AR_DZ: "ar",
        Language.AR_EG: "ar",
        Language.AR_IQ: "ar",
        Language.AR_JO: "ar",
        Language.AR_KW: "ar",
        Language.AR_LB: "ar",
        Language.AR_LY: "ar",
        Language.AR_MA: "ar",
        Language.AR_OM: "ar",
        Language.AR_QA: "ar",
        Language.AR_SA: "ar",
        Language.AR_SY: "ar",
        Language.AR_TN: "ar",
        Language.AR_YE: "ar",
        # Bengali
        Language.BN: "bn",
        Language.BN_BD: "bn",
        Language.BN_IN: "bn",
        # Czech
        Language.CS: "cs",
        Language.CS_CZ: "cs",
        # Danish
        Language.DA: "da",
        Language.DA_DK: "da",
        # German
        Language.DE: "de",
        Language.DE_AT: "de",
        Language.DE_CH: "de",
        Language.DE_DE: "de",
        # Greek
        Language.EL: "el",
        Language.EL_GR: "el",
        # English
        Language.EN: "en",
        Language.EN_AU: "en",
        Language.EN_CA: "en",
        Language.EN_GB: "en",
        Language.EN_HK: "en",
        Language.EN_IE: "en",
        Language.EN_IN: "en",
        Language.EN_KE: "en",
        Language.EN_NG: "en",
        Language.EN_NZ: "en",
        Language.EN_PH: "en",
        Language.EN_SG: "en",
        Language.EN_TZ: "en",
        Language.EN_US: "en",
        Language.EN_ZA: "en",
        # Spanish
        Language.ES: "es",
        Language.ES_AR: "es",
        Language.ES_BO: "es",
        Language.ES_CL: "es",
        Language.ES_CO: "es",
        Language.ES_CR: "es",
        Language.ES_CU: "es",
        Language.ES_DO: "es",
        Language.ES_EC: "es",
        Language.ES_ES: "es",
        Language.ES_GQ: "es",
        Language.ES_GT: "es",
        Language.ES_HN: "es",
        Language.ES_MX: "es",
        Language.ES_NI: "es",
        Language.ES_PA: "es",
        Language.ES_PE: "es",
        Language.ES_PR: "es",
        Language.ES_PY: "es",
        Language.ES_SV: "es",
        Language.ES_US: "es",
        Language.ES_UY: "es",
        Language.ES_VE: "es",
        # Persian
        Language.FA: "fa",
        Language.FA_IR: "fa",
        # Finnish
        Language.FI: "fi",
        Language.FI_FI: "fi",
        # French
        Language.FR: "fr",
        Language.FR_BE: "fr",
        Language.FR_CA: "fr",
        Language.FR_CH: "fr",
        Language.FR_FR: "fr",
        # Hindi
        Language.HI: "hi",
        Language.HI_IN: "hi",
        # Hungarian
        Language.HU: "hu",
        Language.HU_HU: "hu",
        # Indonesian
        Language.ID: "id",
        Language.ID_ID: "id",
        # Italian
        Language.IT: "it",
        Language.IT_IT: "it",
        # Japanese
        Language.JA: "ja",
        Language.JA_JP: "ja",
        # Korean
        Language.KO: "ko",
        Language.KO_KR: "ko",
        # Dutch
        Language.NL: "nl",
        Language.NL_BE: "nl",
        Language.NL_NL: "nl",
        # Polish
        Language.PL: "pl",
        Language.PL_PL: "pl",
        # Portuguese
        Language.PT: "pt",
        Language.PT_BR: "pt",
        Language.PT_PT: "pt",
        # Romanian
        Language.RO: "ro",
        Language.RO_RO: "ro",
        # Russian
        Language.RU: "ru",
        Language.RU_RU: "ru",
        # Slovak
        Language.SK: "sk",
        Language.SK_SK: "sk",
        # Swedish
        Language.SV: "sv",
        Language.SV_SE: "sv",
        # Thai
        Language.TH: "th",
        Language.TH_TH: "th",
        # Turkish
        Language.TR: "tr",
        Language.TR_TR: "tr",
        # Ukrainian
        Language.UK: "uk",
        Language.UK_UA: "uk",
        # Urdu
        Language.UR: "ur",
        Language.UR_IN: "ur",
        Language.UR_PK: "ur",
        # Vietnamese
        Language.VI: "vi",
        Language.VI_VN: "vi",
        # Chinese
        Language.ZH: "zh",
        Language.ZH_CN: "zh",
        Language.ZH_HK: "zh",
        Language.ZH_TW: "zh",
    }
    return language_map.get(language)

class MLXModel(Enum):
    """Class of MLX Whisper model selection options.

    Available models:
        Multilingual models:
            TINY: Smallest multilingual model
            MEDIUM: Good balance for multilingual
            LARGE_V3: Best quality multilingual
            LARGE_V3_TURBO: Finetuned, pruned Whisper large-v3, much faster, slightly lower quality
            DISTIL_LARGE_V3: Fast multilingual
            LARGE_V3_TURBO_Q4: LARGE_V3_TURBO, quantized to Q4
    """

    # Multilingual models
    TINY = "mlx-community/whisper-tiny"
    MEDIUM = "mlx-community/whisper-medium-mlx"
    LARGE_V3 = "mlx-community/whisper-large-v3-mlx"
    LARGE_V3_TURBO = "mlx-community/whisper-large-v3-turbo"
    DISTIL_LARGE_V3 = "mlx-community/distil-whisper-large-v3"
    LARGE_V3_TURBO_Q4 = "mlx-community/whisper-large-v3-turbo-q4"


class WhisperSTTEngineMLX(STTEngine):
    """
    """

    def __init__(
        self,
        *,
        model: str | MLXModel = MLXModel.TINY,
        no_speech_prob: float = 0.6,
        language: Language = Language.EN,
        temperature: float = 0.0,
        **kwargs,
    ):
  
        self._model_name = model if isinstance(model, str) else model.value
        self._no_speech_prob = no_speech_prob
        self._temperature = temperature

        self._settings = {
            "language": language,
        }

        # No need to call _load() as MLX Whisper loads models on demand

    def load(self):
        audio = np.zeros(10, dtype=np.int16)

        # Run the async generator to ensure the model is loaded
        async def _warmup():
            agen = self.run_stt(audio.tobytes())
            try:
                await agen.__anext__()
            except StopAsyncIteration:
                pass
            except Exception:
                pass
              
        loop = asyncio.get_event_loop()
        loop.create_task(_warmup())
      
    def unload(self):
        pass
      
    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert from pipecat Language to Whisper language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            str or None: The corresponding Whisper language code, or None if not supported.
        """
        return language_to_whisper_language(language)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[str, None]:
        """Transcribes given audio using MLX Whisper.

        Args:
            audio: Raw audio bytes in 16-bit PCM format.

        Yields:
            str: Either a TranscriptionFrame containing the transcribed text
                  or an ErrorFrame if transcription fails.

        Note:
            The audio is expected to be 16-bit signed PCM data.
            MLX Whisper will handle the conversion internally.
        """
        try:
            import mlx_whisper

            # Divide by 32768 because we have signed 16-bit data.
            audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

            whisper_lang = self.language_to_service_language(self._settings["language"])
            chunk = await asyncio.to_thread(
                mlx_whisper.transcribe,
                audio_float,
                path_or_hf_repo=self._model_name,
                temperature=self._temperature,
                # language=whisper_lang,
                language=None,
            )
            text: str = ""
            for segment in chunk.get("segments", []):       
                # Drop likely hallucinations
                if segment.get("compression_ratio", None) == 0.5555555555555556:
                    continue

                if segment.get("no_speech_prob", 0.0) < self._no_speech_prob:
                    text += f"{segment.get('text', '')} "
                

            if len(text.strip()) == 0:
                text = None

            if text:
                logger.debug(f"Transcription: [{text}]")
                yield text, self._settings["language"]

        except Exception as e:
            logger.exception(f"MLX Whisper transcription error: {e}")
            yield f"MLX Whisper transcription error: {str(e)}"
