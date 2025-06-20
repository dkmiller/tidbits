import httpx

from http_testing import capture, mock


def test_httpx(tmp_path):
    location = tmp_path / "uuid_generator.yaml"
    # https://www.uuidgenerator.net/api
    url = "https://www.uuidgenerator.net/api/version1"
    state = capture(["httpx"], location)
    response_1 = httpx.get(url)

    state.serialize(location)
    assert location.is_file()

    mock(["httpx"], location)
    response_2 = httpx.get(url)
    # TODO: status code?
    assert response_2.status_code == response_1.status_code
    assert response_2.text == response_1.text
