from contextlib import contextmanager, nullcontext
from inspect import getfile
from functools import wraps
from pathlib import Path
from typing import Literal

from ._httpx import capture as httpx_capture, mock as httpx_mock
from .state import RequestRecorder


SupportedLibraries = Literal["httpx"]


@contextmanager
def capture(libraries: list[SupportedLibraries], location: Path):
    """
    Set up "wrap" / instrumentation logic that will capture all HTTP calls through the
    configured list of libraries, capturing them in the provided location.
    """
    recorder = RequestRecorder()
    if "httpx" in libraries:
        httpx_capture(recorder)

    yield recorder

    recorder.serialize(location)


def mock(libraries: list[SupportedLibraries], location: Path):
    """
    Initialize mocking for the configured list of libraries, loading all HTTP calls
    from the provided location.
    """
    recorder = RequestRecorder.deserialize(location)
    if "httpx" in libraries:
        httpx_mock(recorder)

    return recorder


def _smart_name(func, serialize: str | None):
    func_path = Path(getfile(func))
    serialize = serialize or f"http_testing.{func_path.stem}.{func.__name__}.yaml"
    return func_path.parent / serialize


def mock_http(serialize: str | None = None, overwrite: bool = False):
    """
    The first time this is called it will capture and serialize requests.

    Going forward, forcibly set `overwrite=True` to ensure a new capture.
    """

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            location = _smart_name(func, serialize)
            if overwrite or not location.is_file():
                context = capture(libraries=["httpx"], location=location)
            else:
                context = nullcontext()
                mock(libraries=["httpx"], location=location)

            with context:
                rv = func(*args, **kwargs)

            return rv

        return inner

    return decorator
