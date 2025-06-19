from typing import Any, Callable
from httpx import HTTPTransport
from wrapt import wrap_function_wrapper
from opentelemetry.instrumentation.httpx import _extract_parameters

from .models import Request, Response, RequestResponse

# In httpx >= 0.20.0, handle_request receives a Request object
# TODO: reuse opentelemetry-instrumentation-httpx's _extract_parameters ?

_REQUESTS = []

def _wrapper(wrapped: Callable[..., Any],
        instance: HTTPTransport,
        args: tuple[Any, ...],
        kwargs: dict[str, Any]):
    method, url, headers, stream, extensions = _extract_parameters(
            args, kwargs
        )
    _req = Request(method.decode(), str(url))
    response = wrapped(*args, **kwargs)
    # TODO: handle response.stream (?) + headers.
    response.read()
    _resp = Response(response.status_code, response.text)
    global _REQUESTS
    _REQUESTS.append(RequestResponse(_req, _resp))
    return response
    raise Exception(f"{_req} --> {response} {_resp}")


def capture():
    wrap_function_wrapper("httpx",
            "HTTPTransport.handle_request", _wrapper)
    global _REQUESTS
    return _REQUESTS
