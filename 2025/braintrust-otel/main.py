from contextlib import asynccontextmanager
from dataclasses import dataclass

from dotenv import dotenv_values
from fastapi import FastAPI
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    OpenAIInstrumentor().instrument()

    env = dotenv_values()
    api_key = env["braintrust_api_key"]
    project_id = env["braintrust_project_id"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-bt-parent": f"project_id:{project_id}",
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


app = FastAPI(debug=True, lifespan=lifespan)


@app.post("/chat")
def chat(req: ChatRequest):
    return f"hi, {req.name}"


# Cannot add middleware after the application has started.
FastAPIInstrumentor.instrument_app(app)
