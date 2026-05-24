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
    from deepagents_exploration.agent import agent

    response = await agent.ainvoke(body)  # TODO: better input parsing.
    return {
        "full_response": str(response),
        "last_message": response["messages"][-1].content,
    }
