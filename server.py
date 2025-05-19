import sys
import uvicorn
import os
import argparse
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from contextlib import asynccontextmanager
from loguru import logger
from dotenv import load_dotenv

# from engine.qwen_omni import QwenOmniLLMEngine
# from stt.whisper import WhisperSTTEngine, Model

from engine.mock_engine import MockLLMEngine as QwenOmniLLMEngine
from stt.whisper import WhisperSTTEngineMLX as WhisperSTTEngine, MLXModel as Model

#tts
from tts.openai_tts import OpenAITTSService
from session import RealtimeLLMSession

load_dotenv()

logger.remove()  # Remove existing handlers
logger.add(sys.stderr, level="DEBUG")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # initialize resources
    
    # load whisper before vllm to avoid the later occupy all gpu memory
    stt_engine = WhisperSTTEngine(model=Model.LARGE)
    stt_engine.load()
    app.state.stt_engine = stt_engine

    llm_engine = QwenOmniLLMEngine(model_name="Qwen/Qwen2.5-Omni-3B")
    llm_engine.start()
    app.state.llm_engine = llm_engine
    
    tts_engine = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        voice="alloy",
        model="gpt-4o-mini-tts",
        sample_rate=16000,
    )
    tts_engine.load()
    app.state.tts_engine = tts_engine
    
    logger.info("All engines are ready.")
    
    yield
    # Clean up resources
    stt_engine.unload()
    tts_engine.unload()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/realtime")
async def realtime_api(websocket: WebSocket):
    # create session
    llm_engine = app.state.llm_engine
    stt_engine = app.state.stt_engine
    tts_engine = app.state.tts_engine
    # tts_engine = None
    session = RealtimeLLMSession(websocket, llm_engine, stt_engine, tts_engine)
    main_task = await session.start()
    try:
        await main_task
    except asyncio.CancelledError:
        logger.info("Main task cancelled")
    except Exception as e:
        logger.error(f"Error in main task: {e}")
    finally:
        await session.cleanup()
        logger.info("Session cleaned up")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime LLM Server")
    parser.add_argument(
        "--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host address"
    )
    parser.add_argument("--port", type=int, default=os.getenv("PORT", 7860), help="Port number")
    parser.add_argument("--reload", action="store_true", default=True, help="Reload code on change")

    config = parser.parse_args()

    try:
        import uvicorn

        uvicorn.run("server:app", host=config.host, port=config.port, reload=config.reload)

    except KeyboardInterrupt:
        print("Realtime LLM Server shutting down...")
    
    