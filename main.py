
import sys
import os
import argparse
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
    
    