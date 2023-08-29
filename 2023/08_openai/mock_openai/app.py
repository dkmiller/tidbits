import logging
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from mock_openai.models import Config

log = logging.getLogger("uvicorn")
app = FastAPI()


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


import requests


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


def main():
    # https://stackoverflow.com/a/62856862
    uvicorn.run(
        f"{__name__}:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        # https://www.uvicorn.org/settings/
        reload_dirs=[str(Path(__file__).parent)],
    )
