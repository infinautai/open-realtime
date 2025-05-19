
from typing import Literal, Optional, Union, List, Dict, Any
from pydantic import BaseModel, Field
from utils.id_generator import generateId, RealtimeId
#
# context
#

class ItemContent(BaseModel):
    type: Literal["text", "audio", "input_text", "input_audio"]
    text: Optional[str] = None
    audio: Optional[str] = None  # base64-encoded audio
    transcript: Optional[str] = None
    
    model_config = {
        "arbitrary_types_allowed": True,
    }

class ConversationItem(BaseModel):
    id: Union[str, RealtimeId] = Field(default_factory=lambda: generateId("item"))
    object: Optional[Literal["realtime.item"]] = "realtime.item"
    type: Literal["message", "function_call", "function_call_output"]
    status: Optional[Literal["completed", "in_progress", "incomplete"]] = None
    # role and content are present for message items
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[List[ItemContent]] = None
    # these four fields are present for function_call items
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[str] = None
    
    model_config = {
        "arbitrary_types_allowed": True,
    }


class RealtimeConversation(BaseModel):
    id: Union[str, RealtimeId] = Field(default_factory=lambda: generateId("event"))
    object: Literal["realtime.conversation"]
    
    model_config = {
        "arbitrary_types_allowed": True,
    }

class InboundResponseProperties(BaseModel):
    conversation: Optional[Literal["auto", "none"]] = "auto" # The auto value means that the contents of the response will be added to the default conversation. Set this to none to create an out-of-band response which will not add items to default conversation.
    metadata: Optional[Dict[str, Any]] = None
    input: Optional[List[ConversationItem]] = None
    modalities: Optional[List[Literal["text", "audio"]]] = ["text"]
    instructions: Optional[str] = None
    voice: Optional[str] = None
    output_audio_format: Optional[Literal["pcm16", "g711_ulaw", "g711_alaw"]] = None
    tools: Optional[List[Dict]] = []
    tool_choice: Optional[Literal["auto", "none", "required"]] = None
    temperature: Optional[float] = None
    max_response_output_tokens: Optional[Union[int, Literal["inf"]]] = None
    
    model_config = {
        "arbitrary_types_allowed": True,
    }

class StatusDetails(BaseModel):
    reason: str
    type: Optional[Literal["completed", "cancelled", "failed", "incomplete"]] = None
    error: Optional[Dict[str, Any]] = None
    
    model_config = {
        "arbitrary_types_allowed": True,
    }

class OutboundResponseProperties(BaseModel):
    id: Union[str, RealtimeId] = Field(default_factory=lambda: generateId("resp"))
    conversation_id: Optional[str]
    max_output_tokens: Optional[Union[int, Literal["inf"]]] = None
    metadata: Optional[Dict[str, Any]] = None
    modalities: Optional[List[Literal["text", "audio"]]] = ["text"]
    object: Literal["realtime.response"] = "realtime.response"
    output: Optional[List[ConversationItem]] = None   
    output_audio_format: Optional[Literal["pcm16", "g711_ulaw", "g711_alaw"]] = None
    status: Optional[Literal["completed", "cancelled", "failed", "incomplete"]] = None
    status_details: Optional[StatusDetails] = None
    temperature: Optional[float] = None
    usage: Optional[Dict[str, Any]] = None
    voice: Optional[str] = None
    
    model_config = {
        "arbitrary_types_allowed": True,
    }
    