import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def opentelemetry_init(*, project_id=None, experiment_id=None):
    if project_id is None and experiment_id is None:
        raise ValueError("One of project / experiment ID must exist")

    # https://www.braintrust.dev/docs/guides/traces/integrations#opentelemetry-otel
    if project_id:
        bt_parent = f"project_id:{project_id}"
    else:
        bt_parent = f"experiment_id:{experiment_id}"

    headers = {
        "Authorization": f"Bearer {os.environ['BRAINTRUST_API_KEY']}",
        "x-bt-parent": bt_parent,
    }

    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(headers=headers))
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)
