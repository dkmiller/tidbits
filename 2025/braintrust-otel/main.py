import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from dotenv import dotenv_values
from fastapi import Depends, FastAPI, Request
from openai import AsyncOpenAI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


@dataclass
class ChatRequest:
    name: str


def dotenv() -> dict:
    return dotenv_values()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level="DEBUG")
    OpenAIInstrumentor().instrument()

    # It is not straightforward to leverage FastAPI dependencies within the lifespan:
    # https://github.com/fastapi/fastapi/discussions/11742
    env = dotenv()
    headers = {
        "Authorization": f"Bearer {env['braintrust_api_key']}",
        "x-bt-parent": f"project_id:{env['braintrust_project_id']}",
    }
    print(f"OTel headers = {headers}")

    resource = Resource(attributes={SERVICE_NAME: "braintrust-dan"})
    tracer_provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="https://api.braintrust.dev/otel/v1/traces", headers=headers
        )
    )
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    yield


def openai(env: dict = Depends(dotenv)) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=env["openai_api_key"])


app = FastAPI(debug=True, lifespan=lifespan)


@app.post("/chat")
async def chat(chat_request: ChatRequest, openai: AsyncOpenAI = Depends(openai)):

    # The OTel instrumentation SDK for OpenAI isn't responses-aware yet.
    response = await openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": "Politely great the user by name"},
            {
                "role": "user",
                "content": f"I'm {chat_request.name}",
            },
        ],
    )
    return response.choices[0].message.content


# Cannot add middleware after the application has started.
FastAPIInstrumentor.instrument_app(app)


@app.middleware("http")
async def remove_otel_headers(request: Request, call_next):
    """
    Braintrust does not display traces with external `traceparent` headers. So, we add a middleware
    invoked before the FastAPI OpenTelemetry instrumentor that removes the `traceparent` header
    from the incoming request, following https://stackoverflow.com/a/69934314.
    """
    headers = request.headers.mutablecopy()
    if "traceparent" in headers:
        del headers["traceparent"]
    request._headers = headers
    request.scope.update(headers=request.headers.raw)
    response = await call_next(request)
    return response
