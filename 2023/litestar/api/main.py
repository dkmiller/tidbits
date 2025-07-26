from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from litestar import Litestar, get
from litestar.contrib.opentelemetry import OpenTelemetryConfig

tracer = trace.get_tracer(__name__)


@dataclass
class Health:
    status: bool


@get("/health")
async def health() -> Health:
    return Health(status=True)


@get("/")
async def index() -> str:
    return "Hello, world!"


@get("/books/{book_id:int}")
async def get_book(book_id: int) -> dict[str, int]:
    with tracer.start_as_current_span("book_id"):
        return {"book_id": book_id}


# https://opentelemetry.io/docs/instrumentation/python/exporters/
resource = Resource(attributes={SERVICE_NAME: "api"})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
open_telemetry_config = OpenTelemetryConfig(tracer_provider=provider)


app = Litestar([health, index, get_book], middleware=[open_telemetry_config.middleware])
