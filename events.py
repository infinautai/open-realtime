
import json
from typing import Literal, Optional, Union, List, Dict, Any
from pydantic import BaseModel, Field
from items import (
    ConversationItem,
    RealtimeConversation,
    InboundResponseProperties,
    OutboundResponseProperties,
)
from loguru import logger
from utils.id_generator import generateId, RealtimeId
#
# session properties
#


class InputAudioTranscription(BaseModel):
    """Configuration for audio transcription settings.

    Attributes:
        model: Transcription model to use (e.g., "whisper-tiny", "whisper-1").
        language: Optional language code for transcription.
        prompt: Optional transcription hint text.
    """

    model: str = "whisper-tiny"
    language: Optional[str]
    prompt: Optional[str]

    def __init__(
        self,
        model: Optional[str] = "whisper-tiny",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        super().__init__(model=model, language=language, prompt=prompt)
        # if self.model != "gpt-4o-transcribe" and (self.language or self.prompt):
        #     raise ValueError(
        #         "Fields 'language' and 'prompt' are only supported when model is 'gpt-4o-transcribe'"
        #     )
    
    model_config = {
        "arbitrary_types_allowed": True,
    }

class TurnDetection(BaseModel):
    type: Optional[Literal["server_vad"]] = "server_vad"
    threshold: Optional[float] = 0.5
    prefix_padding_ms: Optional[int] = 200
    silence_duration_ms: Optional[int] = 600
    
    model_config = {
        "arbitrary_types_allowed": True,
    }

# class SemanticTurnDetection(BaseModel):
#     type: Optional[Literal["semantic_vad"]] = "semantic_vad"
#     eagerness: Optional[Literal["low", "medium", "high", "auto"]] = None
#     create_response: Optional[bool] = None
#     interrupt_response: Optional[bool] = None

class SessionProperties(BaseModel):
    modalities: Optional[List[Literal["text"]]] = None #only text is supported
    instructions: Optional[str] = None
    voice: Optional[str] = "alloy"  # TTS voice setting
    tts_model: Optional[str] = "gpt-4o-mini-tts"  # TTS model setting
    input_audio_format: Optional[Literal["pcm16"]] = None # "g711_ulaw", "g711_alaw"
    output_audio_format: Optional[Literal["pcm16"]] = None # "g711_ulaw", "g711_alaw"
    input_audio_transcription: Optional[InputAudioTranscription] = None
    # set turn_detection to False to disable turn detection
    turn_detection: Optional[Union[TurnDetection, bool]] = Field(
        default=None
    )
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Literal["auto", "none", "required"]] = None
    temperature: Optional[float] = None
    max_response_output_tokens: Optional[Union[int, Literal["inf"]]] = None
    
    model_config = {
        "arbitrary_types_allowed": True,
    }


#
# error class
#
class RealtimeError(BaseModel):
    type: str
    code: Optional[str] = ""
    message: str
    param: Optional[str] = None
    event_id: Optional[str] = None
    
    model_config = {
        "arbitrary_types_allowed": True,
    }


#
# client events
#
class ClientEvent(BaseModel):
    event_id: Union[str, RealtimeId] = Field(default_factory=lambda: generateId("event"))
    
    model_config = {
        "arbitrary_types_allowed": True,
    }

class SessionUpdateEvent(ClientEvent):
    type: Literal["session.update"] = "session.update"
    session: SessionProperties

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(*args, **kwargs)

        # Handle turn_detection so that False is serialized as null
        if "turn_detection" in dump["session"]:
            if dump["session"]["turn_detection"] is False:
                dump["session"]["turn_detection"] = None

        return dump


class InputAudioBufferAppendEvent(ClientEvent):
    type: Literal["input_audio_buffer.append"] = "input_audio_buffer.append"
    audio: str  # base64-encoded audio

class InputAudioBufferCommitEvent(ClientEvent):
    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"

class InputAudioBufferClearEvent(ClientEvent):
    type: Literal["input_audio_buffer.clear"] = "input_audio_buffer.clear"

class ConversationItemCreateEvent(ClientEvent):
    type: Literal["conversation.item.create"] = "conversation.item.create"
    previous_item_id: Optional[str] = None
    item: ConversationItem

# Send this event to truncate a previous ASSISTANT message’s audio. 
class ConversationItemTruncateEvent(ClientEvent):
    type: Literal["conversation.item.truncate"] = "conversation.item.truncate"
    item_id: Union[str, RealtimeId]
    content_index: int
    audio_end_ms: int

class ConversationItemDeleteEvent(ClientEvent):
    type: Literal["conversation.item.delete"] = "conversation.item.delete"
    item_id: Union[str, RealtimeId]


class ConversationItemRetrieveEvent(ClientEvent):
    type: Literal["conversation.item.retrieve"] = "conversation.item.retrieve"
    item_id: Union[str, RealtimeId]


class ResponseCreateEvent(ClientEvent):
    type: Literal["response.create"] = "response.create"
    response: Optional[InboundResponseProperties] = None


class ResponseCancelEvent(ClientEvent):
    type: Literal["response.cancel"] = "response.cancel"
    response_id: Optional[Union[str, RealtimeId]] = None

#
# server events
#


class ServerEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: generateId("event"))
    type: str

    model_config = {
        "arbitrary_types_allowed": True,
    }


class SessionCreatedEvent(ServerEvent):
    type: Literal["session.created"] = "session.created"
    session: SessionProperties


class SessionUpdatedEvent(ServerEvent):
    type: Literal["session.updated"] = "session.updated"
    session: SessionProperties


class ConversationCreated(ServerEvent):
    type: Literal["conversation.created"] = "conversation.created"
    conversation: RealtimeConversation


class ConversationItemCreated(ServerEvent):
    type: Literal["conversation.item.created"] = "conversation.item.created"
    previous_item_id: Optional[Union[str, RealtimeId]] = None
    item: ConversationItem


class ConversationItemInputAudioTranscriptionDelta(ServerEvent):
    type: Literal["conversation.item.input_audio_transcription.delta"] = "conversation.item.input_audio_transcription.delta"
    item_id: Union[str, RealtimeId]
    content_index: int
    delta: str


class ConversationItemInputAudioTranscriptionCompleted(ServerEvent):
    type: Literal["conversation.item.input_audio_transcription.completed"] = "conversation.item.input_audio_transcription.completed"
    item_id: Union[str, RealtimeId]
    content_index: int
    transcript: str


class ConversationItemInputAudioTranscriptionFailed(ServerEvent):
    type: Literal["conversation.item.input_audio_transcription.failed"] = "conversation.item.input_audio_transcription.failed"
    item_id: Union[str, RealtimeId]
    content_index: int
    error: RealtimeError


class ConversationItemTruncated(ServerEvent):
    type: Literal["conversation.item.truncated"] = "conversation.item.truncated"
    item_id: Union[str, RealtimeId]
    content_index: int
    audio_end_ms: int


class ConversationItemDeleted(ServerEvent):
    type: Literal["conversation.item.deleted"] = "conversation.item.deleted"
    item_id: Union[str, RealtimeId]


class ConversationItemRetrieved(ServerEvent):
    type: Literal["conversation.item.retrieved"] = "conversation.item.retrieved"
    item: ConversationItem


class ResponseCreated(ServerEvent):
    type: Literal["response.created"] = "response.created"
    response: "OutboundResponseProperties"


class ResponseDone(ServerEvent):
    type: Literal["response.done"] = "response.done"
    response: "OutboundResponseProperties"


class ResponseOutputItemAdded(ServerEvent):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    response_id: Union[str, RealtimeId]
    output_index: int
    item: ConversationItem


class ResponseOutputItemDone(ServerEvent):
    type: Literal["response.output_item.done"] = "response.output_item.done"
    response_id: Union[str, RealtimeId]
    output_index: int
    item: ConversationItem


# class ResponseContentPartAdded(ServerEvent):
#     type: Literal["response.content_part.added"]
#     response_id: Union[str, RealtimeId]
#     item_id: Union[str, RealtimeId]
#     output_index: int
#     content_index: int
#     part: ItemContent


# class ResponseContentPartDone(ServerEvent):
#     type: Literal["response.content_part.done"]
#     response_id: Union[str, RealtimeId]
#     item_id: Union[str, RealtimeId]
#     output_index: int
#     content_index: int
#     part: ItemContent


class ResponseTextDelta(ServerEvent):
    type: Literal["response.text.delta"] = "response.text.delta"
    response_id: Union[str, RealtimeId]
    item_id: Union[str, RealtimeId]
    output_index: int
    content_index: int
    delta: str


class ResponseTextDone(ServerEvent):
    type: Literal["response.text.done"] = "response.text.done"
    response_id: Union[str, RealtimeId]
    item_id: Union[str, RealtimeId]
    output_index: int
    content_index: int
    text: str


class ResponseAudioTranscriptDelta(ServerEvent):
    type: Literal["response.audio_transcript.delta"] = "response.audio_transcript.delta"
    response_id: Union[str, RealtimeId]
    item_id: Union[str, RealtimeId]
    output_index: int
    content_index: int
    delta: str


class ResponseAudioTranscriptDone(ServerEvent):
    type: Literal["response.audio_transcript.done"] = "response.audio_transcript.done"
    response_id: Union[str, RealtimeId]
    item_id: Union[str, RealtimeId]
    output_index: int
    content_index: int
    transcript: str


class ResponseAudioDelta(ServerEvent):
    type: Literal["response.audio.delta"] = "response.audio.delta"
    response_id: Union[str, RealtimeId]
    item_id: Union[str, RealtimeId]
    output_index: int
    content_index: int
    delta: str  # base64-encoded audio


class ResponseAudioDone(ServerEvent):
    type: Literal["response.audio.done"] = "response.audio.done"
    response_id: Union[str, RealtimeId]
    item_id: Union[str, RealtimeId]
    output_index: int
    content_index: int


class ResponseAudioCancel(ServerEvent):
    type: Literal["response.audio.cancel"] = "response.audio.cancel"
    response_id: Union[str, RealtimeId]
    item_id: Optional[Union[str, RealtimeId]]= None
    output_index: int = 0
    content_index: int = 0
    reason: Optional[str] = None


class ResponseFunctionCallArgumentsDelta(ServerEvent):
    type: Literal["response.function_call_arguments.delta"] = "response.function_call_arguments.delta"
    response_id: Union[str, RealtimeId]
    item_id: Union[str, RealtimeId]
    output_index: int
    call_id: Union[str, RealtimeId]
    delta: str


class ResponseFunctionCallArgumentsDone(ServerEvent):
    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"
    response_id: Union[str, RealtimeId]
    item_id: Union[str, RealtimeId]
    output_index: int
    call_id: Union[str, RealtimeId]
    arguments: str


class InputAudioBufferSpeechStarted(ServerEvent):
    type: Literal["input_audio_buffer.speech_started"] = "input_audio_buffer.speech_started"
    audio_start_ms: int
    item_id: Union[str, RealtimeId]


class InputAudioBufferSpeechStopped(ServerEvent):
    type: Literal["input_audio_buffer.speech_stopped"] = "input_audio_buffer.speech_stopped"
    audio_end_ms: int
    item_id: Union[str, RealtimeId]


class InputAudioBufferCommitted(ServerEvent):
    type: Literal["input_audio_buffer.committed"] = "input_audio_buffer.committed"


class InputAudioBufferCleared(ServerEvent):
    type: Literal["input_audio_buffer.cleared"] = "input_audio_buffer.cleared"


class ErrorEvent(ServerEvent):
    type: Literal["error"] = "error"
    error: RealtimeError

_client_event_types = {
    "session.update": SessionUpdateEvent,
    "input_audio_buffer.append": InputAudioBufferAppendEvent,
    "input_audio_buffer.commit": InputAudioBufferCommitEvent,
    "input_audio_buffer.clear": InputAudioBufferClearEvent,
    "conversation.item.create": ConversationItemCreateEvent,
    "conversation.item.truncate": ConversationItemTruncateEvent,
    "conversation.item.delete": ConversationItemDeleteEvent,
    "conversation.item.retrieve": ConversationItemRetrieveEvent,
    "response.create": ResponseCreateEvent,
    "response.cancel": ResponseCancelEvent,
}

def parse_client_event(str):
    try:
        event = json.loads(str)
        event_type = event["type"]
        if event_type not in _client_event_types:
            raise Exception(f"Unimplemented server event type: {event_type}")
        return _client_event_types[event_type].model_validate(event)
    except Exception as e:
        # raise Exception(f"{e} \n\n{str}")
        logger.error(f"Failed to parse client event: {e}")
        return None
