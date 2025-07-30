from opentelemetry.trace import StatusCode

from implementation import failing_method, some_method


def test_failing_method(captrace):
    assert failing_method() == "failed"

    assert StatusCode.ERROR in [s.status.status_code for s in captrace.spans]


def test_method(captrace):
    assert some_method() == "hi"

    assert set(["inner-span", "some-span"]) <= set(s.name for s in captrace.spans)
    assert StatusCode.ERROR not in [s.status.status_code for s in captrace.spans]

    inner_span = [s for s in captrace.spans if s.name == "inner-span"][0]
    assert [e.name for e in inner_span.events] == ["First event", "Second event"]
