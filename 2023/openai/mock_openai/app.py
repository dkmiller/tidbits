import logging
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
import requests
import uvicorn
from aiohttp import ClientSession
from fastapi import FastAPI, Request, Response
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

from mock_openai.aiohttp import SingletonAiohttp
from mock_openai.models import Config, Mode

log = logging.getLogger("uvicorn")

app = FastAPI(on_shutdown=[SingletonAiohttp.close_aiohttp_client])


_config: Config


@app.post("/config")
def set_config(config: Config):
    global _config
    _config = config


@app.post("/openai/deployments/{rest:path}")
def openai_deployments(
    request: Request,
    rest: str,
    body: dict,
):
    # https://tech.clevertap.com/streaming-openai-app-in-python/
    # https://stackoverflow.com/a/57498146
    global _config

    url = urljoin(_config.base, f"/openai/deployments/{rest}")

    headers = {"api-key": _config.api_key}

    stream = bool(body.get("stream"))
    response = requests.post(
        url, params=request.query_params, headers=headers, json=body, stream=stream
    )

    if stream and response.ok:
        return StreamingResponse(response.iter_content(), headers=response.headers)
    else:
        return Response(response.content, response.status_code, response.headers)


@app.post("/stream")
async def stream(request: Request, size: int = 20, sleep: float = 0.5):
    # https://stackoverflow.com/a/75760884
    def fake_stream():
        for _ in range(size):
            yield b"Some data\n"
            time.sleep(sleep)

    return StreamingResponse(
        fake_stream(),
        media_type="text/event-stream",
        headers={"x-content-type-options": "nosniff"},
    )


@app.post("/stream_openai")
async def stream_openai(request: Request):
    def fake_stream():
        response = requests.post(
            "https://*.openai.azure.com/openai/deployments/gpt35turbo/chat/completions?api-version=2023-07-01-preview",
            headers={"api-key": "..."},
            json={
                "messages": [
                    {"role": "system", "content": "You are an assistant"},
                    {
                        "role": "user",
                        "content": "Count from one to one hundred, with the new number on a new line each time.",
                    },
                ],
                "stream": True,
            },
            stream=True,
        )
        for chunk in response.iter_content():
            log.info(chunk)
            # b'data: {"id":"chatcmpl-7sgAN8fuHDBzuP63jQWMEZRFG7moC","object":"chat.completion.chunk","created":1693266155,"model":"gpt-35-turbo","choices":[{"index":0,"finish_reason":null,"delta":{"content":"\\n"},"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}}}],"usage":null}'
            yield chunk
            # yield b"\n"

    return StreamingResponse(
        fake_stream(),
        media_type="text/event-stream",
        headers={"x-content-type-options": "nosniff"},
    )


@app.get("/test-proxy")
async def test_proxy(mode: Mode):
    # ~ 1s
    if mode == Mode.httpx:
        # Works! (http://localhost:8000/test-proxy)
        # With or without the background part.
        # https://github.com/tiangolo/fastapi/issues/1788#issuecomment-1320916419
        client = httpx.AsyncClient()
        request = client.build_request("get", "https://httpbun.com/get")
        response = await client.send(request, stream=True)
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=response.headers,
            background=BackgroundTask(response.aclose),
        )

    # ~ 1s
    if mode == Mode.requests:
        response = requests.get("https://httpbun.com/get", stream=True)
        return StreamingResponse(
            response.iter_content(),
            status_code=response.status_code,
            headers=response.headers,
        )

    # https://github.com/dkmiller/tidbits/blob/master/2022/06-29_azdo-conns/add-to-all-service-connections.py
    # async with ClientSession() as session:
    # https://stackoverflow.com/a/61741161
    # https://stackoverflow.com/a/45174286
    # It looks like the session closes before the response completes.
    # Doing this "properly" is a mess:
    # https://github.com/tiangolo/fastapi/discussions/8301

    # Fails!
    # aiohttp.client_exceptions.ClientConnectionError: Connection closed
    # With or without the background part.
    # With or without auto_decompress=False
    if mode == Mode.aiohttp_naive:
        async with ClientSession() as session:
            async with session.get("https://httpbun.com/get") as response:
                return StreamingResponse(
                    response.content.iter_any(),
                    response.status,
                    response.headers,
                    background=BackgroundTask(response.wait_for_close),
                )

    # Works!
    # ~ 1s
    if mode == Mode.aiohttp_nowith:
        async with ClientSession() as session:
            response = await session.get("https://httpbun.com/get")
            return StreamingResponse(
                response.content.iter_any(),
                response.status,
                response.headers,
                background=BackgroundTask(response.wait_for_close),
            )

    # Works!
    # ~ .3s (2-3x faster than others)
    # https://github.com/raphaelauv/fastAPI-aiohttp-example/
    if mode == Mode.aiohttp_singleton:
        session = SingletonAiohttp.client_session()
        response = await session.get("https://httpbun.com/get")
        return StreamingResponse(
            response.content.iter_any(),
            response.status,
            response.headers,
            background=BackgroundTask(response.wait_for_close),
        )


import typer

cli = typer.Typer()


@cli.command()
def main(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    # https://stackoverflow.com/a/62856862
    uvicorn.run(
        f"{__name__}:app",
        host=host,
        port=port,
        reload=reload,
        # https://www.uvicorn.org/settings/
        reload_dirs=[str(Path(__file__).parent)],
    )
