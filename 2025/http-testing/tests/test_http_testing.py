import time
import httpx

from http_testing import mock_http


@mock_http()
def test_uuid():
    response = httpx.get("https://www.uuidgenerator.net/api/version1")
    response.raise_for_status()
    assert response.text == "4d953abc-4ed1-11f0-9fe2-0242ac120002"


@mock_http()
def test_test_httpx_multiple_calls():
    response_7 = httpx.get("https://www.uuidgenerator.net/api/version7")
    response_4 = httpx.get("https://www.uuidgenerator.net/api/version4")
    response_1 = httpx.get("https://www.uuidgenerator.net/api/version1")

    assert response_1.text == "4d953abc-4ed1-11f0-9fe2-0242ac120002"
    assert response_4.text == "590f9765-0b0b-43d5-81b8-3d5f0d832b03"
    assert response_7.text == "019793de-9ed3-7e03-b5d0-323d1ca3a79a"

    start = time.perf_counter()
    httpx.get("https://httpbun.com/delay/1")
    end = time.perf_counter()
    # Assert: a "real" HTTP request actually happened.
    assert end - start > 1

    start = time.perf_counter()
    httpx.get("https://www.uuidgenerator.net/api/version1/10")
    end = time.perf_counter()
    # Assert: a "real" HTTP request actually happened.
    assert end - start > 0.01
