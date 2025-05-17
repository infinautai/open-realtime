from loguru import logger
import base64
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from typing import Optional, List, Dict, Union
from pydantic import BaseModel, Field
import asyncio
from events import (
    parse_client_event, 
    ClientEvent, 
    SessionUpdateEvent, 
    ConversationCreated,
    RealtimeConversation,
    InputAudioBufferAppendEvent, 
    InputAudioBufferCommitEvent,
    InputAudioBufferClearEvent,
    ConversationItemCreateEvent,
    ConversationItemTruncateEvent,
    ConversationItemDeleteEvent,
    ConversationItemDeleted,
    ConversationItemRetrieveEvent,
    ConversationItemRetrieved,
    ResponseCreateEvent,
    ResponseCreated,
    OutboundResponseProperties,
    ResponseDone,
    ResponseTextDelta,
    ResponseTextDone,
    ResponseCancelEvent,
    SessionProperties,

    SessionCreatedEvent,
    SessionUpdatedEvent,
    InputAudioBufferCommitted,
    InputAudioBufferCleared,
    ConversationItemCreated,
    ConversationItem,
    
    InputAudioBufferSpeechStarted,
    InputAudioBufferSpeechStopped,
    ConversationItemInputAudioTranscriptionCompleted,
    
    TurnDetection,
    InboundResponseProperties
)
from items import (
    ItemContent,
    ConversationItem,
)
from llm_engine import LLMEngine
from audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
from audio.vad.silero import SileroVADAnalyzer
import numpy as np
from utils import generateId
from stt_engine import STTEngine
from pprint import pprint

DEFAULT_SESSION_PROPERTIES = SessionProperties(
    modalities=["text"],
    instructions="You are a helpful assistant.",
    input_audio_format="pcm16",
    output_audio_format="pcm16",
    input_audio_transcription=None,
    turn_detection=False,
    # temperature=0.2,
    # max_response_output_tokens=4096,
)

DEFAULT_VAD_PARAMS: VADParams = VADParams(
    threshold=0.5,
    prefix_padding_ms=200,
    silence_duration_ms=600,
)

def get_sample_rate(format: str) -> int:
    if format == "pcm16":
        return 16000
    elif format == "g711_ulaw":
        return 8000
    elif format == "g711_alaw":
        return 8000
    else:
        raise ValueError(f"Unsupported audio format: {format}")


class InputAudioBuffer(BaseModel): 
    buffer: bytes = Field(default_factory=lambda: b"")
    item_id: str = Field(default_factory=lambda: generateId("item")) 
    def __len__(self):
        return len(self.buffer)
    def append(self, data: bytes):
        self.buffer += data
    
class RealtimeLLMSession:
    def __init__(self,  websocket: WebSocket, llm_engine: LLMEngine, stt_engine: STTEngine = None, settings: SessionProperties = DEFAULT_SESSION_PROPERTIES): 
        self._websocket: WebSocket = websocket
        self._llm_engine = llm_engine
        
        self._receive_task = None
        self._processing_queue = None
        self._processing_task = None
        self._generating_queue = None
        self._generating_task = None
        self._cancel_response = False
        
        self._cancel_repsonse_id = None
        
        self._settings: SessionProperties = settings # type: SessionProperties
        self._monitor_websocket_task = None

        self._messages: Optional[List[ConversationItem]] = []   #TODO: change the name
        self._input_audio_buffer: InputAudioBuffer = InputAudioBuffer()
        # self._output_audio_buffer = bytearray()

        
        self._default_conversation_id = generateId("conv")
        
        self._vad_analyzer = None
        self._current_vad_state = VADState.QUIET
        
        self._stt_engine = stt_engine
        
    async def start(self):
        await self._websocket.accept()
        logger.info(f"{self} WebSocket connection accepted")
        
        self._messages = []  # May load from a database or other source with kind of session id

        await self.send_event(
            SessionCreatedEvent(
                type="session.created",
                session=self._settings
            )
        )

        # Start receiving messages
        if self._receive_task is None:
            self._generating_queue = asyncio.Queue()
            self._generating_task = asyncio.create_task(self._generate_response())
            
            self._processing_queue = asyncio.Queue()
            self._receive_task = asyncio.create_task(self._receive_messages())
            self._processing_task = asyncio.create_task(self._process_messages())

        await self.send_event(
           ConversationCreated(
                type="conversation.created",
                conversation=RealtimeConversation(
                    id=generateId("event"),
                    object="realtime.conversation",
                    conversation={
                        "id": self._default_conversation_id,
                        "object": "realtime.conversation",
                    }
                )
           )
        )

        return self._receive_task
        
    async def stop(self):
        if self._processing_queue:
            self._processing_queue.put_nowait(None)
        if self._generating_queue:
            self._generating_queue.put_nowait(None)
            
        if self._receive_task:
            logger.info("Stopping session...")
            self._cancel_response = True
            self._cancel_repsonse_id = None
            self._receive_task.cancel()
            self._generating_task.cancel()
            self._processing_task.cancel()
            try:
                logger.info("Waiting for generating task to finish...")
                await self._generating_task
                logger.info("Waiting for receiving task to finish...")
                await self._receive_task
                logger.info("Waiting for processing task to finish...")
                await self._processing_task
                logger.info("Session stopped")
            except asyncio.CancelledError:
                pass
            self._receive_task = None
            self._processing_queue = None
            self._processing_task = None
            self._generating_task = None
            self._generating_queue = None

        try:
            await self._websocket.close()
        except (RuntimeError, WebSocketDisconnect) as e:
            if isinstance(e, RuntimeError) and "Unexpected ASGI message 'websocket.close'" in str(e):
                # Already closed, ignore
                pass
            else:
                # For WebSocketDisconnect or other RuntimeErrors, ignore or handle as needed
                pass

    async def cleanup(self):
        await self.stop()
        
    async def _receive_messages(self):
        try:
            async for data in self._websocket.iter_text():
                event = parse_client_event(data)
                if event is None:
                    logger.warning(f"{self} received invalid data: {data}")
                    continue
                await self._processing_queue.put(event)
    
        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")
        logger.info("Receiving task finished")

    async def _process_messages(self):
        while True:
            try:
                event = await self._processing_queue.get()
                if event is None:
                    break
                await self._handle_events(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{self} exception processing data: {e.__class__.__name__} ({e})")
                import traceback
                traceback.print_exc()
        logger.info(f"Processing task finished")
    async def _put_event(self, event: ClientEvent):
        # Put the event in the processing queue
        await self._processing_queue.put(event)
                   
    def _update_vad_analyzer(self, turn_detection: TurnDetection):
       
        if turn_detection:
            sample_rate=get_sample_rate(self._settings.input_audio_format)
            print(f"sample_rate: {sample_rate}")
            threshold = turn_detection.threshold or DEFAULT_VAD_PARAMS.threshold
            prefix_padding_ms = turn_detection.prefix_padding_ms or DEFAULT_VAD_PARAMS.prefix_padding_ms
            silence_duration_ms = turn_detection.silence_duration_ms or DEFAULT_VAD_PARAMS.silence_duration_ms
            params=VADParams(
                min_volume=threshold,
                start_secs=prefix_padding_ms/1000.0,
                stop_secs=silence_duration_ms/1000.0,
            )
            
            if self._vad_analyzer is None:
                self._vad_analyzer = SileroVADAnalyzer(
                    sample_rate=sample_rate,
                    params=params
                )
                self._vad_analyzer.set_sample_rate(sample_rate)
            else:
                self._vad_analyzer.set_params(params)
                self._vad_analyzer.set_sample_rate(sample_rate)
        else:
            if self._vad_analyzer is not None:
                self._vad_analyzer = None
        
        self._current_vad_state = VADState.QUIET        
                
    async def _handle_events(self, event: ClientEvent):
        try:
            if isinstance(event, SessionUpdateEvent):
                await self._handle_session_update(event)
            elif isinstance(event, InputAudioBufferAppendEvent):
                await self._handle_input_audio_buffer_append(event)
            elif isinstance(event, InputAudioBufferCommitEvent):
                await self._handle_input_audio_buffer_commit(event)
            elif isinstance(event, InputAudioBufferClearEvent):
                await self._handle_input_audio_buffer_clear(event)
            elif isinstance(event, ConversationItemCreateEvent):
                await self._handle_conversation_item_create(event)
            elif isinstance(event, ConversationItemTruncateEvent):
                await self._handle_conversation_item_truncate(event)
            elif isinstance(event, ConversationItemDeleteEvent):
                await self._handle_conversation_item_delete(event)
            elif isinstance(event, ConversationItemRetrieveEvent):
                await self._handle_conversation_item_retrieve(event)
            elif isinstance(event, ResponseCreateEvent):
                await self._handle_response_create(event)
            elif isinstance(event, ResponseCancelEvent):
                await self._handle_response_cancel(event)
            else:
                logger.warning(f"{self} received unknown event type: {type(event)}")
        except Exception as e:
            import traceback
            logger.error(f"{self} exception handling event: {e.__class__.__name__} ({e})")
            traceback.print_exc()

    async def _handle_session_update(self, event: SessionUpdateEvent):
        # Handle session update events
        logger.debug(f"Session update event: {event}")
      
        # self._settings = self._settings.model_copy(
        #     update=event.session.model_dump()
        # )
        self._settings = SessionProperties(**event.session.model_dump())
                
        # Update the VAD analyzer if needed
        self._update_vad_analyzer(turn_detection=self._settings.turn_detection)
    
        await self.send_event(
            SessionUpdatedEvent(session=self._settings)
        )

    async def _handle_input_audio_buffer_append(self, event: InputAudioBufferAppendEvent):
        # logger.debug(f"Input audio buffer append event: {event}")
        
        curent_length = len(self._input_audio_buffer)
        # convert base64 to bytes
        audio = base64.b64decode(event.audio)
        self._input_audio_buffer.append(audio)
        
        commit_buffer = False
        if self._vad_analyzer:
            # Analyze the audio buffer for VAD
            vad_states = []
            for (new_vad_state, time_offset) in self._vad_analyzer.analyze_audio(audio, curent_length):
                if new_vad_state != VADState.STARTING and new_vad_state != VADState.STOPPING:
                    vad_states.append((new_vad_state, time_offset))
        
            if len(vad_states) > 0:
                new_vad_state, time_offset = vad_states[-1]
                if ( new_vad_state != self._current_vad_state ):
                    if new_vad_state == VADState.SPEAKING:
                        logger.debug(f"Speech started, time_offset: {time_offset}")
                        await self.send_event(
                            InputAudioBufferSpeechStarted(
                                audio_start_ms=int(time_offset * 1000),
                                item_id=self._input_audio_buffer.item_id,
                            )
                        )
                        
                        self._cancel_response = True
                        self._cancel_repsonse_id = None
                        
                    elif new_vad_state == VADState.QUIET:
                        logger.debug(f"Speech stopped, time_offset: {time_offset}")
                        commit_buffer = True
                        await self.send_event(
                            InputAudioBufferSpeechStopped(
                                audio_end_ms=int(time_offset * 1000),
                                item_id=self._input_audio_buffer.item_id,
                            )
                        )
                        
                        self._cancel_response = False
                        self._cancel_repsonse_id = None

                    self._current_vad_state = new_vad_state
                    
        if commit_buffer:
            buffer = self._input_audio_buffer.buffer #save for stt
            conv_item = await self._handle_input_audio_buffer_commit(
                InputAudioBufferCommitEvent(
                    type="input_audio_buffer.commit",
                    item_id=self._input_audio_buffer.item_id,
                )
            )    
            # trigger response generation
            await self._put_event(ResponseCreateEvent(type="response.create"))
            
            if self._stt_engine:
                # Transcribe the audio buffer
                transcript = ""
                async for text, language in self._stt_engine.run_stt(buffer):
                    if text:
                        print(f"Transcription: [{text}]")
                        transcript += text
                        await self.send_event(ConversationItemInputAudioTranscriptionCompleted(
                            transcript=text,
                            item_id=self._input_audio_buffer.item_id,
                            content_index=0
                        ))
                        await asyncio.sleep(0)
                if transcript and conv_item.content[0].type == "input_audio":
                    conv_item.content[0].transcript = transcript
                    
    async def _handle_input_audio_buffer_commit(self, event: InputAudioBufferCommitEvent)-> ConversationItem:
        logger.debug(f"Input audio buffer commit event: {event}")
        
        # Process the audio buffer by creating conversation item
        item = ConversationItem(
            object="realtime.item",
            type="message",
            role="user",
            content=[ItemContent(
                type="input_audio",
                audio=base64.b64encode(self._input_audio_buffer.buffer),
            )]
        )
       
        self._messages.append(item)
        await self.send_event(
            InputAudioBufferCommitted(
                type="input_audio_buffer.committed",
            )
        )
        self._input_audio_buffer = InputAudioBuffer()  # Clear the buffer after commit
        return item

    async def _handle_input_audio_buffer_clear(self, event: InputAudioBufferClearEvent):
        logger.debug(f"Input audio buffer clear event: {event}")
        # Clear the audio buffer
        self._input_audio_buffer = InputAudioBuffer()
        await self.send_event(
            InputAudioBufferCleared(type="input_audio_buffer.cleared")
        )

    async def _handle_conversation_item_create(self, event: ConversationItemCreateEvent):
        logger.debug(f"Conversation item create event: {event}")

        new_item = event.item
        previous_id = event.previous_item_id
        insert_index = None

        if previous_id:
            for idx, item in enumerate(self._messages):
                if item.id == previous_id:
                    insert_index = idx + 1
                    break

        if insert_index is not None:
            self._messages.insert(insert_index, new_item)
        else:
            previous_id = self._messages[-1].id if self._messages else None
            self._messages.append(new_item)

        await self.send_event(
            ConversationItemCreated(
                type="conversation.item.created",
                previous_item_id=previous_id,
                item=new_item,
            )
        )

    async def _handle_conversation_item_truncate(self, event: ConversationItemTruncateEvent):
        """Truncate a previous assistant message's audio"""
        logger.debug(f"Conversation item truncate event: {event}")
        

    async def _handle_conversation_item_delete(self, event: ConversationItemDeleteEvent):
        # Handle conversation item delete events
        logger.debug(f"Conversation item delete event: {event}")

        item_id = event.item_id
        self._messages = [
            item for item in self._messages if item.id != item_id
        ]

        await self.send_event(
            ConversationItemDeleted(
                type="conversation.item.deleted",
                item_id=event.item_id,
            )
        )

    async def _handle_conversation_item_retrieve(self, event: ConversationItemRetrieveEvent):
        # Handle conversation item retrieve events
        logger.debug(f"Conversation item retrieve event: {event}")

        item_id = event.item_id
        for item in self._messages:
            if item.id == item_id:
                await self.send_event(
                    ConversationItemRetrieved(
                        type="conversation.item.retrieved",
                        item=item,
                    )
                )
                break

    async def _handle_response_create(self, event: ResponseCreateEvent):
        await self._generating_queue.put(event)
    
    async def _do_generate(self, event: ResponseCreateEvent):
        try:
            item_id = generateId("item")
            response_id = generateId("resp")

            in_response = event.response
            conv_id = self._default_conversation_id 
            if not in_response or in_response.conversation != "auto":
                # Create a new conversation ID
                conv_id = generateId("conv")

            out_response = OutboundResponseProperties(
                id=response_id,
                conversation_id=conv_id,
                object="realtime.response",
                metadata=in_response.metadata if in_response else None,
                modalities=['text'],
                temperature=self._settings.temperature,
                max_output_tokens=self._settings.max_response_output_tokens,
            )
            if in_response and in_response.temperature is not None:
                out_response.temperature = in_response.temperature
            if in_response and in_response.max_response_output_tokens is not None:
                out_response.max_output_tokens = in_response.max_response_output_tokens
            
            #Construct the input messages
            conversation_items = [
                ConversationItem(
                    object="realtime.item",
                    type="message",
                    role="system",
                    content=[
                        ItemContent(
                            type="text",
                            text=in_response.instructions or self._settings.instructions if in_response else self._settings.instructions,
                        )
                    ]
                )    
            ]
            out_of_band = False
            if in_response and in_response.input:
                for item in in_response.input:
                    conversation_items.append(item)
            elif in_response and in_response.conversation == "none": 
                # Create an out-of-band response
                out_of_band = True
            else:
                # Use the default conversation
                conversation_items.extend(self._messages)
            
            generated_text = ""
            messages = self.convert_to_listofdict(conversation_items)
            async for (delta, finished) in self._llm_engine.generate_response(
                messages,
                temperature=out_response.temperature,
                max_tokens=out_response.max_output_tokens,
                request_id=item_id,
            ):
                if self._cancel_response:
                    if self._cancel_repsonse_id and self._cancel_repsonse_id != response_id:
                        logger.info(f"Failed to cancel response: {self._cancel_repsonse_id}")
                        self._cancel_response = False
                        self._cancel_repsonse_id = None
                    else:
                        logger.info(f"Response generation cancelled: {response_id}")
                        out_response.status = "cancelled"
                        await self.send_event(ResponseDone(
                                type="response.done",
                                response=out_response
                            ))
                        self._cancel_response = False
                        self._cancel_repsonse_id = None
                        return

                generated_text += delta
                await self.send_event(
                    ResponseTextDelta(
                        delta=delta,
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                    )
                )
                await asyncio.sleep(0)

            await self.send_event(
                ResponseTextDone(
                    type="response.text.done",
                    text=generated_text,
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0, # same as the index of input[item] in ResponseDone
                    content_index=0, # index of the content in the item as following ConversationItem
                )
            )

            item = ConversationItem(
                id=item_id,
                type="message",
                role="assistant",
                content=[
                    ItemContent(
                        type="text",
                        text=generated_text,
                    )
                ]
            )
            if not out_of_band:
                self._messages.append(item)
            
            out_response.output = [item]
            out_response.status = "completed"
            await self.send_event(ResponseDone(type="response.done", response=out_response))
        except asyncio.CancelledError:
            logger.info("Response generation cancelled")
            out_response.status = "cancelled"
            await self.send_event(ResponseDone(
                    type="response.done",
                    response=out_response
            ))
        except Exception as e:
            import traceback
            logger.error(f"Error during response generation: {e}")
            traceback.print_exc()
            out_response.status = "failed"
            await self.send_event(ResponseDone(
                    type="response.done",
                    response=out_response
                ))

    async def _generate_response(self):
        while True:
            try:
                event = await self._generating_queue.get()
                if event is None:
                    break
                await self._do_generate(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{self} exception generating response: {e.__class__.__name__} ({e})")
                import traceback
                traceback.print_exc()
        logger.info(f"Generating task finished")

    async def _handle_response_cancel(self, event: ResponseCancelEvent):
        # Handle response cancel events
        logger.debug(f"Response cancel event: {event}")
        self._cancel_response = True
        self._cancel_repsonse_id = event.response_id
        

    async def send_event(self, event: ClientEvent):
        data = event.model_dump_json(exclude_none=True)
        try:
            await self._websocket.send_text(data)
        except (RuntimeError, WebSocketDisconnect) as e:
            logger.error(f"Exception sending data: {e.__class__.__name__} ({e})")
            # Optionally set a flag here to prevent further sends
            pass
        
    def convert_to_listofdict(self, conv_items: list[ConversationItem])-> list[Dict[str, Union[str, List[Dict[str, str]]]]]:
        """Format chat messages into a prompt for the model.

        Args:
            messages: List of ConversationItem objects representing the chat history.

        Returns:
            List of dictionaries representing the formatted messages.
        """
        messages = []
        for item in conv_items:
            contents = []
            for part in item.content:
                content = {
                    "type": part.type
                }
                if part.type == "text" or part.type == "input_text":
                    content["type"] = "text" 
                    content["text"] = part.text
                elif part.type == "input_audio":  #no audio which is for output, fillter it out
                    if part.transcript:
                        content["type"] = "text" 
                        content["text"] = part.transcript
                    else:
                        content["type"] = "audio" 
                        audio = base64.b64decode(part.audio)
                        #convert to np.array
                        audio_data = np.frombuffer(audio, dtype=np.int16)
                        content["audio"] = audio_data
            
                contents.append(content)

            messages.append({
                "role": item.role,
                "content": contents
            })
    
        return messages

