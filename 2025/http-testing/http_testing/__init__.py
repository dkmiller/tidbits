from pathlib import Path
from typing import Literal

from ._httpx import capture as httpx_capture


SupportedLibraries = Literal["httpx"]


def capture(libraries: list[SupportedLibraries], location: Path):
    """
    Set up "wrap" / instrumentation logic that will capture all HTTP calls through the
    configured list of libraries, capturing them in the provided location.
    """
    if "httpx" in libraries:
        requests = httpx_capture()

    # TODO: context manager so we know when to "dump" things.


def mock(libraries: list[SupportedLibraries], location: Path):
    """
    Initialize mocking for the configured list of libraries, loading all HTTP calls
    from the provided location.
    """


def mock_http(serialize: str, overwrite: bool = False):
    location = Path(serialize)
    if overwrite:
        capture(libraries=["httpx"], location=location)
    else:
        mock(libraries=["httpx"], location=location)
