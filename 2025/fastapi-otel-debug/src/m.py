from opentelemetry import trace

tracer = trace.get_tracer(__name__)


def foo():
    with tracer.start_as_current_span("foo") as span:
        rv = "foo_value2"
        span.set_attribute("foo.rv", rv)
        return rv
