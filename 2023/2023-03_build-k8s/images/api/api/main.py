import logging

from aiohttp import ClientSession
from fastapi import FastAPI
import uvicorn

from api import models

# TODO: why?
log = logging.getLogger("uvicorn")

app = FastAPI()


@app.get("/")
async def root():
    log.info("Hi from the root!")
    return {"message": "Hello World"}


@app.get("/cat-fact")
async def cat_fact() -> models.CatFact:
    """
    Follow https://apipheny.io/free-api/ to get free facts about cats.
    """
    # TODO: dependency injection for sessions:
    # https://github.com/tiangolo/fastapi/discussions/8301
    async with ClientSession() as session:
        async with session.get("https://catfact.ninja/fact") as response:
            parsed = await response.json()
            log.info("Got a cat fact!")
            return models.CatFact(**parsed)


@app.get("/health")
async def health() -> models.Health:
    # https://emmer.dev/blog/writing-meaningful-health-check-endpoints/
    return models.Health(ok=True)


def main():
    # https://stackoverflow.com/a/62856862
    uvicorn.run(app, host="0.0.0.0", port=8000)
