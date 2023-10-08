from typing import Any, Dict

import requests
from fastapi import Depends, FastAPI
from fastapi_utils.tasks import repeat_every
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

_VALUES = {}


@tracer.start_as_current_span("refresh_url_value")
def _refresh(url):
    global _VALUES
    span = trace.get_current_span()
    span.set_attribute("refresh.url", url)
    response = requests.get(url)
    _VALUES[url] = response.json()


def setup(app: FastAPI):
    @app.on_event("startup")
    @repeat_every(seconds=10)
    def refresh_values() -> None:
        global _VALUES
        for url in list(_VALUES.keys()):
            _refresh(url)


@tracer.start_as_current_span("get_url_value")
def _config_value(url: str):
    span = trace.get_current_span()
    span.set_attribute("get.url", url)
    global _VALUES
    if url not in _VALUES:
        _refresh(url)
    return _VALUES[url]


def config(url: str) -> Dict[str, Any]:
    return Depends(lambda: _config_value(url))
