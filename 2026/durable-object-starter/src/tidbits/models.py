from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class BeginResponse(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, validate_by_name=True)

    challenge_id: str
    options: dict


class Credential(BaseModel):
    id: str  # base64url credential ID
    user_id: str
    public_key: bytes  # COSE key
    sign_count: int
    transports: list[str]
    aaguid: str | None = None
    backed_up: bool = False
    device_type: str = "single_device"
    nickname: str | None = None


ChallengeKind = Literal["registration", "authentication"]


class PendingChallenge(BaseModel):
    id: str
    challenge: bytes
    user_id: int
    kind: ChallengeKind
    expires_at: datetime


class RelyingParty(BaseModel):
    id: str  # "example.com" — registrable domain, no scheme, no port
    name: str  # "Acme"
    origin: str  # "https://example.com" — full origin the browser sees


class WebauthnUser(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, validate_by_name=True)

    username: str
    display_name: str


class User(WebauthnUser):
    id: int
