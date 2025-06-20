from inspect import getfile
from functools import wraps
from pathlib import Path
from typing import Literal

from ._httpx import capture as httpx_capture, mock as httpx_mock
from .state import RequestRecorder


SupportedLibraries = Literal["httpx"]


def capture(libraries: list[SupportedLibraries], location: Path):
    """
    Set up "wrap" / instrumentation logic that will capture all HTTP calls through the
    configured list of libraries, capturing them in the provided location.
    """
    recorder = RequestRecorder()
    if "httpx" in libraries:
        httpx_capture(recorder)

    return recorder

    # TODO: context manager so we know when to "dump" things.


def mock(libraries: list[SupportedLibraries], location: Path):
    """
    Initialize mocking for the configured list of libraries, loading all HTTP calls
    from the provided location.
    """
    recorder = RequestRecorder.deserialize(location)
    if "httpx" in libraries:
        httpx_mock(recorder)

    return recorder


def mock_http(serialize: str, overwrite: bool = False):
    """
    The first time this is called it will capture and serialize requests.

    Going forward, forcibly set `overwrite=True` to ensure a new capture.
    """

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            # TODO: "smart" file naming via method name.
            location = Path(getfile(func)).parent / serialize
            if overwrite or not location.is_file():
                recorder = capture(libraries=["httpx"], location=location)
            else:
                mock(libraries=["httpx"], location=location)
                recorder = None
            try:
                rv = func(*args, **kwargs)
            finally:
                if recorder:
                    recorder.serialize(location)
            return rv

        return inner

    return decorator
