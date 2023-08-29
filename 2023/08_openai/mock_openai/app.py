import logging
from pathlib import Path
from typing import Annotated
import requests

# import openai
import uvicorn
from fastapi import FastAPI, Query, Request, Response, HTTPException
from fastapi.responses import StreamingResponse


log = logging.getLogger("uvicorn")
app = FastAPI()


# def _stream():
#     # https://tech.clevertap.com/streaming-openai-app-in-python/
#     import requests
#     response = requests.post("https://air-openai-test2.openai.azure.com/openai/deployments/gpt35turbo/chat/completions?api-version=2023-07-01-preview", headers={"api-key": "..."}, json={"messages": [{"role": "system", "content": "You are an assistant"}, {"role": "user", "content": "Count from one to one hundred, with the new number on a new line each time."}], "stream": True, "top_p": .95, "temperature": .7, "max_tokens": 1000}, stream=True)
#     for line in response.iter_lines():
#         # log.info(line)
#         yield line
    # return response.iter_lines()

# for line in response.iter_lines():
#     print(line)

#     try:
#         response = openai.ChatCompletion.create(engine="gpt4", **body)
#         for chunk in response:
#             yield chunk["choices"][0]["delta"].get("content", "")
#     except Exception as e:
#         log.error("Error: %s", e)
#         return 503




def _stream__(response: requests.Response):
    # https://tech.clevertap.com/streaming-openai-app-in-python/
    for chunk in response.iter_content():
        yield chunk
    # for line in response.iter_lines():
    #     # log.info(line)
    #     yield line
    #     yield b"\n"




@app.post("/openai/deployments/{rest:path}")
def openai_deployments(
    request: Request,
    rest: str,
    body: dict,
):
#     # log.info("Headers: %s", request.headers)
    stream = bool(body.get("stream"))
#     # from urllib.parse import urljoin

#     # import openai
#     # openai.api_base = "https://air-openai-test2.openai.azure.com"
#     # openai.api_key = "..."
#     # openai.api_type = "azure"
#     # openai.api_version = api_version

#     # url = urljoin(
#     #     "https://air-openai-test2.openai.azure.com/openai/deployments/", rest
#     # )
#     # import requests

#     #     # from copy import deepcopy
#     # headers = {}
#     # for key in ["accept-encoding", "content-type"]:
#     #     headers[key] = request.headers[key]
#     # # headers["content-type"] = request.headers["content-type"]
#     # # headers["accept-encoding"] = request.headers["accept-encoding"]
#     # # headers = request.headers.mutablecopy()
#     # headers["api-key"] = "..."
#     # params = {"api-version": api_version}
#     # # log.info("Request URL: %s", url)
#     # # log.info("Request body: %s", body)
#     # # log.info("Request headers: %s", headers)
#     # # log.info("Request parameters: %s", params)
    response = requests.post(f"https://air-openai-test2.openai.azure.com/openai/deployments/{rest}", params=request.query_params, headers={"api-key": "..."}, json=body, stream=stream)
    log.info("Response status: %s", response.status_code)
    response.raise_for_status()
    if stream:
        log.info("Returning streaming response take 1")
        # https://tech.clevertap.com/streaming-openai-app-in-python/
        # return response.iter_lines()
        return StreamingResponse(response.iter_content(), media_type="text/event-stream")#, headers={'x-content-type-options': 'nosniff'})
    else:
        return response.json()




#         # def _stream():
#         #     # https://tech.clevertap.com/streaming-openai-app-in-python/
#         #     for line in response.iter_lines():
#         #         # log.info(line)
#         #         yield line + b"\n"
#         response = requests.post(url, headers={"api-key": "..."}, params=params, json=body, stream=True)

#         return StreamingResponse(_stream__(response), media_type="text/event-stream")#, headers={'x-content-type-options': 'nosniff'})
#     else:
#         raise HTTPException(status_code=501, detail={"message": "not supported yet"})

#     # This doesn't actually stream :/
#     # https://stackoverflow.com/a/57498146
#     session = requests.Session()
#     with session.post(url, headers=headers, json=body, params=params, stream=stream) as response:
#     # response = requests.post(url, json=body, headers=headers, params=params, stream=stream)
#         log.info("Response status: %s", response.status_code)
#         log.info("Response headers: %s", response.headers)
#         # log.info("Response text: %s", response.text)
#         response.raise_for_status()
#         if stream:
#             log.info("Streaming response")
#             return StreamingResponse(content=response.iter_lines(), status_code=response.status_code, headers=response.headers, media_type="text/event-stream")
#         return Response(content=response.content, status_code=response.status_code, headers=response.headers)
#     # return response


# import time

@app.post("/stream")
async def stream(request: Request, size: int=20, sleep: float = 0.5):
    # https://stackoverflow.com/a/75760884
    def fake_stream():
        for _ in range(size):
            yield b"Some data\n"
            time.sleep(sleep)

    return StreamingResponse(fake_stream(), media_type="text/event-stream", headers={'x-content-type-options': 'nosniff'})

import requests



@app.post("/stream_openai")
async def stream_openai(request: Request):
    def fake_stream():
        response = requests.post("https://air-openai-test2.openai.azure.com/openai/deployments/gpt35turbo/chat/completions?api-version=2023-07-01-preview", headers={"api-key": "..."}, json={"messages": [{"role": "system", "content": "You are an assistant"}, {"role": "user", "content": "Count from one to one hundred, with the new number on a new line each time."}], "stream": True}, stream=True)
        for chunk in response.iter_content():
            log.info(chunk)
            # b'data: {"id":"chatcmpl-7sgAN8fuHDBzuP63jQWMEZRFG7moC","object":"chat.completion.chunk","created":1693266155,"model":"gpt-35-turbo","choices":[{"index":0,"finish_reason":null,"delta":{"content":"\\n"},"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}}}],"usage":null}'
            yield chunk
            # yield b"\n"

    return StreamingResponse(fake_stream(), media_type="text/event-stream", headers={'x-content-type-options': 'nosniff'})




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

