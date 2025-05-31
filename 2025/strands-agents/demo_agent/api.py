from dataclasses import dataclass
from fastapi import Depends, FastAPI
from strands import Agent

from demo_agent.dependencies import primes_agent, tracer


@dataclass
class PromptRequest:
    prompt: str


app = FastAPI(title="Strands agents demo")


# TODO: how to get structured agent outputs?
@app.post("/primes_agent")
def call_primes_agent(
    request: PromptRequest,
    agent: Agent = Depends(primes_agent),
    # Forcibly enable tracing. TODO: do this via middleware etc.?
    _=Depends(tracer),
):
    response = agent(request.prompt)
    return {"result": str(response)}
