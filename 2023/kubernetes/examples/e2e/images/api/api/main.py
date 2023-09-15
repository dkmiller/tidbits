import logging

import uvicorn
from aiohttp import ClientSession
from api import builders, models
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi_injector import Injected, attach_injector
from opentelemetry import trace
from starlette.background import BackgroundTask

# TODO: why?
log = logging.getLogger("uvicorn")
tracer = trace.get_tracer(__name__)

app = FastAPI()
attach_injector(app, builders.injector())


@app.get("/")
async def root():
    log.info("Hi from the root!")
    return {"message": "Hello World"}


@app.get("/cat-fact")
async def cat_fact(session=Injected(ClientSession)) -> models.CatFact:
    """
    Follow https://apipheny.io/free-api/ to get free facts about cats.
    """
    with tracer.start_as_current_span("cat_fact"):
        async with session.get("https://catfact.ninja/fact") as response:
            parsed = await response.json()
            log.info("Got a cat fact!")
            return models.CatFact(**parsed)


@app.get("/health")
async def health() -> models.Health:
    # https://emmer.dev/blog/writing-meaningful-health-check-endpoints/
    return models.Health(ok=True)


@app.get("/httpbun")
async def httpbun(session=Injected(ClientSession)):
    response = await session.get("https://httpbun.com/get")
    return StreamingResponse(
        response.content.iter_any(),
        response.status,
        response.headers,
        background=BackgroundTask(response.wait_for_close),
    )


def main():
    # https://stackoverflow.com/a/62856862
    uvicorn.run(app, host="0.0.0.0", port=8000)
