import sys
import uvicorn
import os
import argparse
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from contextlib import asynccontextmanager
from loguru import logger
# from engine.qwen_omni import QwenOmniLLMEngine
from engine.mock_engine import MockLLMEngine as QwenOmniLLMEngine
from stt.whisper import WhisperSTTEngineMLX as WhisperSTTEngine, MLXModel
from session import RealtimeLLMSession

logger.remove()  # Remove existing handlers
logger.add(sys.stderr, level="DEBUG")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # initialize resources
    llm_engine = QwenOmniLLMEngine()
    llm_engine.start()
    app.state.llm_engine = llm_engine

    stt_engine = WhisperSTTEngine(model=MLXModel.LARGE_V3)
    stt_engine.load()
    app.state.stt_engine = stt_engine
    
    yield
    # Clean up resources
    stt_engine.unload()

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
    session = RealtimeLLMSession(websocket, llm_engine, stt_engine)
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
    
    