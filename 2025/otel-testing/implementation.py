from opentelemetry import trace

tracer = trace.get_tracer(__name__)


@tracer.start_as_current_span("some-span")
def some_method():
    with tracer.start_as_current_span("inner-span") as span:
        span.add_event("First event")
        span.add_event("Second event")

    return "hi"


def failing_method():
    try:
        with tracer.start_as_current_span("failing-span"):
            raise RuntimeError("failure!")
    except RuntimeError:
        return "fail silently"
