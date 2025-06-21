# HTTP mocking/testing in Python

``` python
import httpx
from http_testing import mock_http


@mock_http()
def some_test_involving_httpx():
    httpx.get("https://foo.org", auth=SomeAuth())
    pass
```

## Future

- [ ] Async
- [ ] Streaming
- [ ] aiohttp
- [ ] requests

## Development

``` bash
ruff check --fix . && ruff format .

pytest
```

## Links

- [opentelemetry-instrumentation-httpx / `__init__.py`](https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/instrumentation/opentelemetry-instrumentation-httpx/src/opentelemetry/instrumentation/httpx/__init__.py)
- [Wrapt](https://wrapt.readthedocs.io/en/master/wrappers.html#function-wrappers)
- [PyPI &gt; opentelemetry-instrumentation-httpx](https://pypi.org/project/opentelemetry-instrumentation-httpx/)
- [pytest-httpx](https://colin-b.github.io/pytest_httpx/)
- [pytest-httpx / `__init__.py`](https://github.com/Colin-b/pytest_httpx/blob/develop/pytest_httpx/__init__.py)
