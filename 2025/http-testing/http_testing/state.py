from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from .models import Request, Response, RequestResponse


@dataclass
class RequestRecorder:
    _state: list[RequestResponse] = field(default_factory=list)

    def record(self, request: Request, response: Response):
        self._state.append(RequestResponse(request, response))

    def serialize(self, path: Path):
        y_ = OmegaConf.to_yaml({"requests": self._state})
        path.write_text(y_)

    @classmethod
    def deserialize(cls, path: Path) -> RequestRecorder:
        raise NotImplementedError()

    def mock(self, request: Request) -> Response | None:
        pass
