from dotenv import dotenv_values
from fastapi import Depends
from strands import Agent
from strands_tools import calculator, current_time, python_repl
from strands.types.models import Model
from strands.models.openai import OpenAIModel
from strands.telemetry.tracer import get_tracer
from strands.telemetry import Tracer

from demo_agent.tools import nth_prime


def dotenv() -> dict[str, str]:
    return dotenv_values()  # type: ignore


def tracer(dotenv: dict[str, str] = Depends(dotenv)) -> Tracer:
    return get_tracer(
        service_name="dan-strands-agents",
        otlp_endpoint="https://api.honeycomb.io",
        otlp_headers={"x-honeycomb-team": dotenv["honeycomb_api_key"]},
        enable_console_export=True,
    )


def model(dotenv: dict[str, str] = Depends(dotenv)) -> Model:
    return OpenAIModel(
        client_args={
            "api_key": dotenv["openai_api_key"],
        },
        model_id="gpt-4o",
        params={"max_tokens": 1000},
    )


def tools() -> list:
    return [calculator, current_time, python_repl, nth_prime]


# Via dependency injection, the returned agent can depend on arbitrary information
# from the incoming HTTP request.
def primes_agent(model: Model = Depends(model), tools: list = Depends(tools)) -> Agent:
    return Agent(
        model=model,
        system_prompt="""
You are an AI agent who helps with information and calculations about prime numbers.
        """,
        tools=tools,
    )
