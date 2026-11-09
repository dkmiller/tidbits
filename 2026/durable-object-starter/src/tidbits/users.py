import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from durable import MyDurableObject
from tidbits.models import (
    ChallengeKind,
    Credential,
    PendingChallenge,
    User,
    WebauthnUser,
)

CHALLENGE_TTL = timedelta(seconds=60)


@dataclass
class UserStore:
    durable: MyDurableObject

    async def create_user(self, user: WebauthnUser) -> User:
        # We shouldn't have to await...
        arr = await self.durable.sql(
            "INSERT INTO users_v3 (user_name, display_name) VALUES (?, ?) RETURNING user_id",
            user.username,
            user.display_name
        )  # type: ignore

        return User(
            **user.model_dump(),
            id=arr[0]["user_id"]
        )

    # def get_user(self, user_id: str) -> Optional[User]: ...
    async def get_user_by_username(self, username: str) -> User | None:
        # It's bonkers that we have to await here.
        arr = await self.durable.sql("SELECT * FROM users_v3 WHERE user_name = ?", username)  # type: ignore

        if not arr:
            return None

        first = arr[0]

        return User(
            id=first["user_id"],
            username=first["user_name"],
            display_name=first["display_name"],
        )

    async def credentials_for_user(self, user_id: int) -> Sequence[Credential]:
        # We shouldn't have to await...
        arr = await self.durable.sql("SELECT * FROM credentials_v3 WHERE user_id = ?", user_id)  # type: ignore
        return list(map(Credential.model_validate, arr))

    # def get_credential(self, credential_id: str) -> Optional[Credential]: ...
    # def add_credential(self, credential: Credential) -> None: ...
    # def update_sign_count(self, credential_id: str, sign_count: int) -> None: ...


@dataclass
class ChallengeStore:
    durable: MyDurableObject

    async def put(self, challenge: PendingChallenge) -> None:
        await self.durable.sql(
            "INSERT INTO challenges_v3 (id, challenge, user_id, kind, expires_at) VALUES (?, ?, ?, ?, ?)",
            challenge.id,
            challenge.challenge,
            challenge.user_id,
            challenge.kind,
            int(challenge.expires_at.timestamp()),
        ) # type: ignore

    async def take(self, challenge_id: str) -> PendingChallenge | None:
        """Fetch and delete in one step. Single use, always."""
        ...

    async def store(self, challenge: bytes, *, user_id: int, kind: ChallengeKind) -> str:
        challenge_id = secrets.token_urlsafe(32)
        await self.put(
            PendingChallenge(
                id=challenge_id,
                challenge=challenge,
                user_id=user_id,
                kind=kind,
                expires_at=datetime.now(UTC) + CHALLENGE_TTL
            )
        )

        return challenge_id
