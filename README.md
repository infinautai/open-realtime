# OpenRealtime API Service - Real-time Voice Language Model Server

A WebSocket-based real-time voice conversation system that supports voice input, intelligent text generation, and voice output for complete conversational workflows.

## 🎯 Project Background

In the past half year, omni-modal models and speech-to-speech models such as Qwen Omni, Ming-Omni, Qwen Audio and Kimi Audio have been released one after another, highlighting that omni-modal models are currently a major focus in AI research. However, there is still a lack of open-source tools for real-time API interaction.

This project is inspired by the OpenAI realtime API and playground, and aims to provide an open-source version of a realtime playground. We hope this tool can help accelerate the adoption and development of these advanced models in the community.

## ✨ Features

- 🎤 **Real-time Voice Input**: Supports streaming audio data processing and real-time voice activity detection (VAD)
- 🤖 **Intelligent Conversation**: Integrates large language models with support for multimodal input (text + audio)
- 🔊 **Voice Synthesis**: Supports multiple TTS services (OpenAI TTS, ElevenLabs)
- 🌐 **WebSocket Real-time Communication**: Low-latency bidirectional communication with event-driven architecture
- 📝 **Speech-to-Text**: High-precision speech recognition based on Whisper models
- 🔄 **Streaming Processing**: Supports streaming generation of text and audio, optimizing response latency
- 🎛️ **Configurable**: Supports multiple audio formats, sample rates, and model parameter adjustments

## 🏗️ System Architecture

```
┌─────────────────┐    WebSocket    ┌──────────────────┐
│   Client App    │ ◄────────────► │   FastAPI Server │
└─────────────────┘                └──────────────────┘
                                           │
                   ┌────────────or─────────┼────────────or───────────┐
                   │                        │                        │
           ┌───────▼────────┐    ┌─────────▼────────┐    ┌──────────▼─────────┐
           │   STT Engine   │    │    LLM Engine    │    │     TTS Engine     │
           │   (Whisper)    │    │  (Qwen2.5-Omni) │    │ (OpenAI/ElevenLabs)│
           └────────────────┘    └──────────────────┘    └────────────────────┘
                   │                        │                        │
           ┌───────▼────────┐    ┌─────────▼────────┐    ┌──────────▼─────────┐
           │ Speech Recognition│  │  Text Generation │    │  Speech Synthesis  │
           └────────────────┘    └──────────────────┘    └────────────────────┘
```




## 🚀 Quick Start

### Requirements

- Python 3.12+
- macOS/Linux (recommended)
- Sufficient memory for loading models

### Install Dependencies

```bash
# Clone the project
git clone <repository-url>
cd realtime_llm

# Install dependencies
pip install -r requirements.txt

# Or use uv (recommended)
uv sync
```

### Environment Configuration

Create a `.env` file:

```bash
# OpenAI API configuration (for TTS)
OPENAI_API_KEY=your_openai_api_key

# ElevenLabs API configuration (optional)
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# Server configuration
HOST=0.0.0.0
PORT=7866
```

### Start the Server

```bash
# Start with default configuration
python main.py

# Or specify parameters
python main.py --host 0.0.0.0 --port 7866

# Development mode (auto-reload)
python main.py --reload
```

After the server starts, the WebSocket endpoint is: `ws://localhost:7866/realtime`

## 🖥️ Client Example

We provide a complete web client example that can interact directly with this server:

**🔗 Client Project**: [Realtime Playground](https://github.com/infinautai/realtime-playground)

The client provides:
- 🎤 Real-time voice recording and playback
- 💬 WebSocket connection management
- 🎛️ Real-time parameter adjustment interface
- 📊 Audio visualization
- 🔧 Complete debugging tools

## 📖 Usage Guide

### WebSocket Event System

The client communicates with the server through WebSocket by sending JSON-formatted events:

#### Client Events

- `session.update` - Update session configuration
- `input_audio_buffer.append` - Add audio data
- `input_audio_buffer.commit` - Commit audio buffer
- `input_audio_buffer.clear` - Clear audio buffer
- `conversation.item.create` - Create conversation item
- `response.create` - Request response generation
- `response.cancel` - Cancel response generation

#### Server Events

- `session.created` - Session created successfully
- `session.updated` - Session configuration updated
- `input_audio_buffer.speech_started` - Speech detected started
- `input_audio_buffer.speech_stopped` - Speech detected stopped
- `response.text.delta` - Text response delta
- `response.audio.delta` - Audio response delta
- `response.done` - Response generation completed

### Supported Audio Formats

- **PCM16** (16kHz, 16-bit)
- **G711 μ-law** (8kHz)
- **G711 A-law** (8kHz)

### Example: Basic Conversation Flow

```javascript
// 1. Establish WebSocket connection
const ws = new WebSocket('ws://localhost:7866/realtime');

// 2. Update session configuration
ws.send(JSON.stringify({
    type: 'session.update',
    session: {
        modalities: ['text', 'audio'],
        instructions: 'You are a friendly AI assistant',
        turn_detection: {
            type: 'server_vad',
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 800
        }
    }
}));

// 3. Send audio data
ws.send(JSON.stringify({
    type: 'input_audio_buffer.append',
    audio: base64AudioData
}));

// 4. Listen for responses
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received event:', data.type);
};
```

## 🔧 Configuration Options

### Session Configuration

```json
{
    "modalities": ["text", "audio"],
    "instructions": "System prompt",
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "input_audio_transcription": {
        "model": "whisper-1"
    },
    "turn_detection": {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 800
    },
    "temperature": 0.8,
    "max_response_output_tokens": 4096
}
```

### VAD Parameter Tuning

- `threshold`: Voice detection threshold (0.0-1.0)
- `prefix_padding_ms`: Pre-speech buffer time
- `silence_duration_ms`: Silence detection duration

### TTS Engine Configuration

**OpenAI TTS:**
```json
{
    "voice": "alloy",  // alloy, echo, fable, onyx, nova, shimmer
    "model": "gpt-4o-mini-tts",
    "sample_rate": 16000
}
```

**ElevenLabs:**
```json
{
    "voice": "voice_id",
    "model": "eleven_multilingual_v2",
    "sample_rate": 16000
}
```

## 🏃‍♂️ Development and Testing

### Run Tests

```bash
# TTS engine tests
python tts/test_tts_engine.py

# Unit tests
python -m pytest utils/test_*.py
```

### Development Mode

The project supports hot-reload development:

```bash
python main.py --reload
```

### Mock Mode

For development and testing without real models:

```python
# In server.py
from engine.mock_engine import MockLLMEngine as QwenOmniLLMEngine
```

## 📁 Project Structure

```
realtime_llm/
├── audio/                   # Audio processing modules
│   ├── resamplers/         # Audio resamplers
│   └── vad/               # Voice activity detection
├── engine/                 # LLM engine implementations
│   ├── mock_engine.py     # Mock engine (for testing)
│   └── qwen_omni.py       # Qwen multimodal engine
├── stt/                   # Speech-to-text
│   └── whisper.py         # Whisper STT implementation
├── tts/                   # Text-to-speech
│   ├── openai_tts.py      # OpenAI TTS service
│   └── elevenlabs_tts.py  # ElevenLabs TTS service
├── utils/                 # Utility functions
├── events.py              # Event definitions
├── session.py             # Session management
├── server.py              # FastAPI server
└── main.py               # Application entry point
```

## 🔧 Deployment

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 7866

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "7866"]
```

### Production Environment Recommendations

- Use reverse proxy (Nginx) to handle WebSocket connections
- Configure appropriate resource limits
- Enable logging and monitoring
- Use process managers (systemd, supervisord)

## 🤝 Contributing

Issues and Pull Requests are welcome!

### Development Guidelines

- Use type annotations
- Follow PEP 8 code style
- Add appropriate logging
- Write test cases

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**Q: Voice detection is inaccurate?**
A: Adjust VAD parameters, especially `threshold` and `silence_duration_ms`

**Q: Poor audio quality?**
A: Check sample rate settings, ensure client and server sample rates match

**Q: High response latency?**
A: Optimize network connection, consider using faster TTS services or local deployment

**Q: High memory usage?**
A: Adjust model parameters, use smaller models or increase system memory

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

📧 For questions, please submit an Issue or contact the maintainers.