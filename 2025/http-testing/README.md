# HTTP mocking/testing in Python

``` python
import httpx
from http_testing import mock_http


@mock_http(serialize="test_name.yaml", overwrite=True)
def some_test_involving_httpx():
    httpx.get("https://foo.org", auth=SomeAuth())
    pass
```

## Future

- [ ] aiohttp
- [ ] requests

## Links

- https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/instrumentation/opentelemetry-instrumentation-httpx/src/opentelemetry/instrumentation/httpx/__init__.py
- https://wrapt.readthedocs.io/en/master/wrappers.html#function-wrappers
- https://pypi.org/project/opentelemetry-instrumentation-httpx/
- https://colin-b.github.io/pytest_httpx/
- https://github.com/Colin-b/pytest_httpx/blob/develop/pytest_httpx/__init__.py
