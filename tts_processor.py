import asyncio
import base64
from dataclasses import dataclass
from enum import Enum
from events import ResponseAudioDelta, ResponseAudioDone, ResponseAudioCancel
from loguru import logger
from utils.string import match_endofsentence
from utils.id_generator import RealtimeId, generateId
from typing import Union, Optional


class TTSAction(Enum):
    CREATE = "create"
    CANCEL = "cancel"

@dataclass
class TTSInputEvent:
    text: str
    audio: bytes = None
    isDelta: bool = False
    response_id: Union[str, RealtimeId] = None
    item_id: Union[str, RealtimeId] = None
    action: TTSAction = TTSAction.CREATE

class TTSProcessor:
    def __init__(self, session, tts_engine, params=None):
        self.session = session  
        self.tts_engine = tts_engine
        # Initialize TTS parameters with default values
        self.params = {
            'voice': 'alloy',  # Default OpenAI voice
            'model': 'gpt-4o-mini-tts',  # Default TTS model
            'sample_rate': 16000,  # Default sample rate
            'instructions': None,  # Optional TTS instructions
        }
        if params:
            self.params.update(params)
        self._tts_queue = asyncio.Queue()
        self._cancel_response_id = generateId("resp", 0)
        self._worker_task = None
        self._response_text_buffer = ""

    async def update_params(self, **kwargs):
        """Update TTS parameters and validate them.
        
        Args:
            **kwargs: Parameters to update, including:
                voice (str): One of 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'
                model (str): TTS model name
                sample_rate (int): Audio sample rate in Hz
                instructions (str, optional): TTS instructions/prompt
        """
        valid_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        
        if 'voice' in kwargs:
            voice = kwargs['voice']
            if voice not in valid_voices:
                raise ValueError(f"Invalid voice '{voice}'. Must be one of {valid_voices}")
        
        if 'sample_rate' in kwargs:
            sample_rate = kwargs['sample_rate']
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                raise ValueError(f"Invalid sample rate: {sample_rate}")
        
        self.params.update(kwargs)
        
        # If this is an OpenAI TTS engine, apply the updates directly
        if hasattr(self.tts_engine, 'set_voice'):
            await self.tts_engine.set_voice(self.params['voice'])
        if hasattr(self.tts_engine, 'set_model'):
            await self.tts_engine.set_model(self.params['model'])

    async def start(self):
        if not self._worker_task:
            self._worker_task = asyncio.create_task(self._tts_worker())

    async def stop(self):
        if self._worker_task:
            # Cancel any ongoing TTS synthesis
            await self._cancel_active_synthesis(reason="tts_processor_stopping")
            # Signal worker to stop and wait for it to finish
            await self._tts_queue.put(None)
            await self._worker_task
            self._worker_task = None
            logger.info("TTS processor stopped")

    async def push_event(self, event: TTSInputEvent):
        if event.audio and len(event.audio) > 0:
            # If audio is provided, send it directly
            if event.isDelta:
                await self.session.send_event(
                    ResponseAudioDelta(
                        delta=base64.b64encode(event.audio).decode('utf-8'),
                        response_id=event.response_id,
                        item_id=event.item_id,
                        output_index=0,
                        content_index=0,
                    )
                )
            else:
                # If it's not a delta, we assume it's a complete audio response
                await self.session.send_event(
                    ResponseAudioDone(
                        response_id=event.response_id,
                        item_id=event.item_id,
                        output_index=0,
                        content_index=0,
                    )
                )
            return
        
        # Handle cancellation events
        if event.action == TTSAction.CANCEL:
            self._response_text_buffer = ""
            # Directly cancel any active synthesis with this response_id
            await self._cancel_active_synthesis(
                response_id=event.response_id, 
                reason="explicit_cancel_action"
            )
            # Also push to queue to ensure any pending events are cancelled
            await self._tts_queue.put(event)
            return

        # Handle TTS events
        if event.action == TTSAction.CREATE:
            if not event.isDelta:
                # flush the last text buffer
                if self._response_text_buffer:
                    # Create a new TTS event for the buffered text
                    tts_event = TTSInputEvent(
                        text=self._response_text_buffer,
                        audio=event.audio,
                        isDelta=False,
                        response_id=event.response_id,
                        item_id=event.item_id,
                        action=TTSAction.CREATE
                    )
                    await self._tts_queue.put(tts_event)
                    self._response_text_buffer = ""
                return
            
            # buffer the text until we find an end-of-sentence marker
            delta = event.text
            self._response_text_buffer += delta
            text_line = None
            eos_end_marker = match_endofsentence(self._response_text_buffer)
            if eos_end_marker:
                text_line = self._response_text_buffer[:eos_end_marker]
                self._response_text_buffer = self._response_text_buffer[eos_end_marker:]
            
            if not text_line:
                return

            # Create new TTS event for the complete sentence
            tts_event = TTSInputEvent(
                text=text_line,
                audio=event.audio,
                isDelta=event.isDelta,
                response_id=event.response_id,
                item_id=event.item_id,
                action=TTSAction.CREATE
            )
            await self._tts_queue.put(tts_event)

    async def _tts_worker(self):
        while True:
            event = await self._tts_queue.get()
            if event is None:
                break
            
            # Skip processing events for responses that have been cancelled
            if event.action == TTSAction.CREATE and event.response_id and event.response_id <= self._cancel_response_id:
                logger.info(f"Skipping TTS event for cancelled response: {event.response_id}")
                continue
            
            # Handle cancel events by ignoring them in the worker
            # (the cancellation was already handled in push_event)
            if event.action == TTSAction.CANCEL:
                logger.info(f"Received cancel event in worker queue for response: {event.response_id}")
                continue
                
            await self._do_tts(event)
        logger.info("TTS task finished")

    async def _do_tts(self, event: TTSInputEvent):
        try:
            async for audio in self.tts_engine.run_tts(event.text, **self.params):
                if not audio or len(audio) == 0:
                    continue
                
                # Check for cancellation
                if event.response_id <= self._cancel_response_id:
                    logger.info(f"Cancelling TTS synthesis for response: {event.response_id}, cancelled by: {self._cancel_response_id}")
                    return
                
                # Send audio delta
                await self.session.send_event(
                    ResponseAudioDelta(
                        delta=base64.b64encode(audio).decode('utf-8'),
                        response_id=event.response_id,
                        item_id=event.item_id,
                        output_index=0,
                        content_index=0,
                    )
                )
               
            
            # Send final done event after all chunks are processed
            await self.session.send_event(
                ResponseAudioDone(
                    response_id=event.response_id,
                    item_id=event.item_id,
                    output_index=0,
                    content_index=0,
                )
            )
            logger.info(f"Finished TTS for text: {event.text}")
        except Exception as e:
            logger.error(f"Error during TTS processing: {str(e)}")
            # You might want to send an error event to the client here

    async def _cancel_active_synthesis(self, response_id=None, reason="tts_cancelled"):
        """Cancel the currently active TTS synthesis.
        
        Args:
            response_id: Optional specific response ID to cancel. If None, cancels any active synthesis.
            reason: Reason for cancellation, used for logging.
        
        Returns:
            bool: True if cancellation was requested, False otherwise.
        """
        # Set cancellation flags
        self._cancel_response_id = response_id or generateId("resp")
        
        # Clear the text buffer to prevent partial sentence processing
        self._response_text_buffer = ""
        
        await self.session.send_event(
            ResponseAudioCancel(
                response_id=self._cancel_response_id,
                reason=reason,
            )
        )
        
        logger.info(f"TTS synthesis cancellation requested: ID={response_id or 'any'}, reason={reason}")
        
        # We can't immediately know if cancellation succeeded
        # The actual cancellation happens in _do_tts
        return True
