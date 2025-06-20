from functools import partial
from typing import Any, Callable

from httpx import HTTPTransport, Response as HttpxResponse
from wrapt import wrap_function_wrapper
from opentelemetry.instrumentation.httpx import _extract_parameters

from .models import Request, Response
from .state import RequestRecorder


def capture_request(args, kwargs) -> Request:
    method, url, headers, stream, extensions = _extract_parameters(args, kwargs)
    return Request(method.decode(), str(url))


def _capture_wrapper(
    wrapped: Callable[..., Any],
    instance: HTTPTransport,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    __recorder: RequestRecorder,
):
    captured_request = capture_request(args, kwargs)
    response = wrapped(*args, **kwargs)
    # TODO: handle response.stream / bytes (?) + headers.
    response.read()
    captured_response = Response(response.status_code, response.text)
    __recorder.record(captured_request, captured_response)
    return response


def capture(recorder: RequestRecorder):
    wrap_function_wrapper(
        "httpx",
        "HTTPTransport.handle_request",
        partial(_capture_wrapper, __recorder=recorder),
    )


def _mock_wrapper(
    wrapped: Callable[..., Any],
    instance: HTTPTransport,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    __recorder: RequestRecorder,
) -> HttpxResponse:
    captured_request = capture_request(args, kwargs)
    if mocked_response := __recorder.mock(captured_request):
        return HttpxResponse(
            status_code=mocked_response.status, text=mocked_response.text
        )
    return wrapped(*args, **kwargs)


def mock(recorder: RequestRecorder):
    wrap_function_wrapper(
        "httpx",
        "HTTPTransport.handle_request",
        partial(_mock_wrapper, __recorder=recorder),
    )
