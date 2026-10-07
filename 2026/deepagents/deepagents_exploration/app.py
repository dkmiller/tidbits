from contextlib import asynccontextmanager
from fastapi import FastAPI

from deepagents_exploration.config import initialize_environment


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_environment()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/invoke")
async def invoke(body: dict):
    # DeepAgents picks up environment variables on import not invocation, so we must
    # import the agent after the FastAPI lifespan has initialized the environment.
    from deepagents_exploration.agent import agent

    response = await agent.ainvoke(body)  # TODO: better input parsing.
    return response  # TODO: does this even work?
