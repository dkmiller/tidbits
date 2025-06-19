import httpx

from http_testing import capture, mock


def test_httpx(tmp_path):
    location = tmp_path / "uuid_generator.yaml"
    # https://www.uuidgenerator.net/api
    url = "https://www.uuidgenerator.net/api/version1"
    capture(["httpx"], location)
    response_1 = httpx.get(url)

    from http_testing._httpx import _REQUESTS
    from omegaconf import OmegaConf
    y_ = OmegaConf.to_yaml({"requests": _REQUESTS})


    raise Exception(y_)
    assert location.is_file()

    mock(["httpx"], location)
    response_2 = httpx.get(url)
    # TODO: status code?
    assert response_2.text == response_1.text
