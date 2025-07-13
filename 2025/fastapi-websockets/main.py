from dotenv import dotenv_values
from fastapi import Depends, FastAPI

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi_proxy_lib.core.websocket import ReverseWebSocketProxy
from httpx import AsyncClient
from starlette.websockets import WebSocket


def dotenv() -> dict[str, str]:
    return dotenv_values() # type: ignore


# ?api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview
# base_url="ws://echo.websocket.events/"
base_url = "wss://eastus2.api.cognitive.microsoft.com/"
proxy = ReverseWebSocketProxy(
    AsyncClient(headers={"api-key": dotenv()["api_key"]}), base_url=base_url)

@asynccontextmanager
async def close_proxy_event(_: FastAPI) -> AsyncIterator[None]:
    """Close proxy."""
    yield
    await proxy.aclose()

app = FastAPI(lifespan=close_proxy_event)

@app.websocket("/experimental{path:path}")
async def _(websocket: WebSocket, path: str = "", env: dict = Depends(dotenv)):
    # raise Exception(path)
    # raise Exception(env)
    return await proxy.proxy(websocket=websocket, path=path)

# Connecting to wss://eastus2.api.cognitive.microsoft.com/openai/realtime?api-version=2025-04-01-preview&deployment=gpt-4o-realtime-preview