import os
from fastapi import FastAPI

from deepagents_exploration.agent import agent


# TODO: configuration management...
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "tidbits-deepagents"


app = FastAPI()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/invoke")
async def invoke(body: dict):
    response = await agent.ainvoke(body) # TODO: better input parsing.
    return {"response": str(response)}
