from functools import partial
from typing import Any, Callable

from httpx import HTTPTransport
from wrapt import wrap_function_wrapper
from opentelemetry.instrumentation.httpx import _extract_parameters

from .models import Request, Response
from .state import RequestRecorder

# In httpx >= 0.20.0, handle_request receives a Request object
# TODO: reuse opentelemetry-instrumentation-httpx's _extract_parameters ?

_RECORDER = RequestRecorder()


# TODO: accept "request state container" as parameter, use partial() below.
def _wrapper(
    wrapped: Callable[..., Any],
    instance: HTTPTransport,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    __recorder: RequestRecorder,
):
    method, url, headers, stream, extensions = _extract_parameters(args, kwargs)
    _req = Request(method.decode(), str(url))
    response = wrapped(*args, **kwargs)
    # TODO: handle response.stream (?) + headers.
    response.read()
    _resp = Response(response.status_code, response.text)
    __recorder.record(_req, _resp)
    return response


def capture(recorder: RequestRecorder):
    wrap_function_wrapper(
        "httpx", "HTTPTransport.handle_request", partial(_wrapper, __recorder=recorder)
    )
