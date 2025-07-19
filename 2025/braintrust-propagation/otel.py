import os


from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def opentelemetry_init(project_id):
    headers = {
        "Authorization": f"Bearer {os.environ['BRAINTRUST_API_KEY']}",
        "x-bt-parent": f"project_id:{project_id}",
    }

    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="https://api.braintrust.dev/otel/v1/traces", headers=headers
        )
    )
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)
