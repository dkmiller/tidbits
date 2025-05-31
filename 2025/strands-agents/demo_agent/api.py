from dataclasses import dataclass
from fastapi import FastAPI

from demo_agent.agent import agent


@dataclass
class PromptRequest:
    prompt: str


app = FastAPI(title="Strands agents demo")


@app.post("/some_agent")
def some_agent(request: PromptRequest):
    _agent = agent
    response = _agent(request.prompt)
    return {
        "content": str(response)
    }
