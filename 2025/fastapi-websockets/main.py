from dotenv import dotenv_values
from fastapi import Depends, FastAPI
from fastapi_proxy_lib.core.websocket import ReverseWebSocketProxy
from httpx import AsyncClient
from starlette.websockets import WebSocket


def dotenv() -> dict[str, str]:
    return dotenv_values()  # type: ignore


app = FastAPI()


async def websocket_proxy(env: dict = Depends(dotenv)):
    base_url = "wss://eastus2.api.cognitive.microsoft.com/"
    rv = ReverseWebSocketProxy(
        AsyncClient(headers={"api-key": dotenv()["api_key"]}), base_url=base_url
    )
    try:
        yield rv
    finally:
        await rv.aclose()


@app.websocket("/experimental{path:path}")
async def experimental_websocket(
    websocket: WebSocket, path: str, proxy=Depends(websocket_proxy)
):
    return await proxy.proxy(websocket=websocket, path=path)


# TODO: figure out a middleware to strip incoming api-key headers:
# https://stackoverflow.com/a/69934314.
