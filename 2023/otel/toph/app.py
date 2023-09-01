from random import randint

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

app = FastAPI()


def fail(value: str):
    # This part captures exceptions nicely.
    with tracer.start_as_current_span("fail"):
        raise Exception(f"Fail: {value}")


def roll_sum(sides, rolls):
    sum = 0
    with tracer.start_as_current_span("roll_sum"):
        for _ in range(0, rolls):
            result = randint(1, sides)
            sum += result
    return str(sum)


@app.get("/fail")
def get_fail(value: str):
    fail(value)


@app.get("/roll")
def roll(sides: int, rolls: int):
    return roll_sum(sides, rolls)
