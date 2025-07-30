from dataclasses import dataclass, field

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from pytest import fixture


@dataclass
class CaptureExporter(SpanExporter):
    spans: list = field(default_factory=list)

    def export(self, spans):
        self.spans.extend(spans)


@fixture
def captrace():
    exporter = CaptureExporter()
    processor = SimpleSpanProcessor(exporter)

    # An existing OpenTelemetry framework (e.g. pytest-opentelemetry) is not in
    # play.
    if not isinstance(provider := trace.get_tracer_provider(), TracerProvider):
        resource = Resource(attributes={})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

    provider.add_span_processor(processor)

    try:
        yield exporter
    finally:
        provider.shutdown()
