from opentelemetry import trace

from implementation import failing_method, some_method


def test_method(captrace):
    some_method()

    assert set(["inner-span", "some-span"]) <= set(s.name for s in captrace.spans)
    assert trace.StatusCode.ERROR not in [s.status.status_code for s in captrace.spans]

    inner_span = [s for s in captrace.spans if s.name == "inner-span"][0]
    assert [e.name for e in inner_span.events] == ["First event", "Second event"]


def test_failing_method(captrace):
    failing_method()

    assert trace.StatusCode.ERROR in [s.status.status_code for s in captrace.spans]
