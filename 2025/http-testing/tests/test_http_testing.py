import httpx

from http_testing import mock_http


@mock_http("uuid_generator.yaml")  # , overwrite=True)
def test_uuid():
    response = httpx.get("https://www.uuidgenerator.net/api/version1")
    response.raise_for_status()
    assert response.text == "6eb0a80e-4d63-11f0-9fe2-0242ac120002"
