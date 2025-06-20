import time
import httpx

from http_testing import mock_http


@mock_http("uuid_generator.yaml")  # , overwrite=True)
def test_uuid():
    response = httpx.get("https://www.uuidgenerator.net/api/version1")
    response.raise_for_status()
    assert response.text == "6eb0a80e-4d63-11f0-9fe2-0242ac120002"


@mock_http("uuid_generator_multiple.yaml")
def test_test_httpx_multiple_calls():
    response_7 = httpx.get("https://www.uuidgenerator.net/api/version7")
    response_4 = httpx.get("https://www.uuidgenerator.net/api/version4")
    response_1 = httpx.get("https://www.uuidgenerator.net/api/version1")

    assert response_1.text == "6eb0a80e-4d63-11f0-9fe2-0242ac120002"
    assert response_4.text == "f0513af1-1cf1-4bfe-b2f2-dc46d8e47267"
    assert response_7.text == "01978a80-7a0a-7301-b845-abba9d5e8d3d"

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
