from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from .models import Request, Response, RequestResponse


@dataclass
class RequestRecorder:
    state: list[RequestResponse] = field(default_factory=list)

    def record(self, request: Request, response: Response):
        self.state.append(RequestResponse(request, response))

    def serialize(self, path: Path):
        y_ = OmegaConf.to_yaml(self)
        path.write_text(y_)

    @classmethod
    def deserialize(cls, path: Path) -> RequestRecorder:
        schema = OmegaConf.structured(cls)
        conf = OmegaConf.load(path)
        merged = OmegaConf.merge(schema, conf)
        return OmegaConf.to_object(merged)  # type: ignore

    def mock(self, request: Request) -> Response | None:
        responses = [x for x in self.state if x.request == request]
        if responses:
            return responses[0].response
