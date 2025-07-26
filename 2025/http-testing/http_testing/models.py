from dataclasses import dataclass


@dataclass
class Request:
    method: str
    url: str


@dataclass
class Response:
    status: int
    text: str


@dataclass
class RequestResponse:
    request: Request
    response: Response
