import json
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status

from durable import MyDurableObject
from tidbits.authentication import begin_registration
from tidbits.models import BeginResponse, RelyingParty, WebauthnUser
from tidbits.users import ChallengeStore, UserStore

app = FastAPI()


def my_durable_object(request: Request) -> MyDurableObject:
    env = request.scope["env"]
    stub = env.MY_DURABLE_OBJECT.getByName(request.url.path)
    return stub


MyObject = Annotated[MyDurableObject, Depends(my_durable_object)]
"""
Resolve a Durable Object instance for an HTTP request.
"""


@app.get("/hi/{name}")
async def say_hi(name: str, my_obj: MyObject):
    greeting = await my_obj.say_hello(name)

    return {"message": str(greeting), "name": name}


def user_store(my_object: MyObject):
    return UserStore(durable=my_object)


User = Annotated[UserStore, Depends(user_store)]


def get_rp(request: Request) -> RelyingParty:
    return RelyingParty(
        id=request.url.hostname or "unknown",
        name="Tidbits",
        origin=str(request.base_url),
    )


RP = Annotated[RelyingParty, Depends(get_rp)]


def get_challenges(my_object: MyObject) -> ChallengeStore:
    return ChallengeStore(durable=my_object)


Challenges = Annotated[ChallengeStore, Depends(get_challenges)]


router = APIRouter(prefix="/auth")

# TODO: rate limits to avoid "crawling" this to find existing usernames.
@router.post("/register/begin")
async def register_begin(user: WebauthnUser, user_store: User, rp: RP, challenges: Challenges):
    """
    Handle the creation of new accounts and addition of passkeys to existing
    accounts.
    """
    existing = await user_store.get_user_by_username(user.username)

    if existing is None:
        target = await user_store.create_user(user)
    elif user == existing.id:
        target = existing
    else:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "That username is taken. Sign in to add another passkey.",
        )

    challenge_id, options_json = await begin_registration(
        rp=rp, user=target, user_store=user_store, challenges=challenges
    )

    return BeginResponse(challenge_id=challenge_id, options=json.loads(options_json))



# @router.post("/register/begin", response_model=BeginResponse)
# def register_begin(
#     body: RegisterBeginRequest,
#     rp: RP,
#     credentials: Credentials,
#     challenges: Challenges,
#     user: MaybeUser,
# ) -> BeginResponse:
#     existing = credentials.get_user_by_username(body.username)

#     if existing is None:
#         target = credentials.create_user(
#             username=body.username,
#             display_name=body.displayName or body.username,
#         )
#     elif user is not None and user.id == existing.id:
#         target = existing
#     else:
#         # Taking over an existing account requires proving you already control
#         # it. Returning 409 does leak that the username exists, which is hard
#         # to avoid on a signup form; rate limit this endpoint accordingly.
#         raise HTTPException(
#             status.HTTP_409_CONFLICT,
#             "That username is taken. Sign in to add another passkey.",
#         )

#     challenge_id, options_json = begin_registration(
#         rp=rp, user=target, credentials=credentials, challenges=challenges
#     )
#     return BeginResponse(challengeId=challenge_id, options=json.loads(options_json))


app.include_router(router)










# """
# HTTP surface for the WebAuthn ceremonies.

# This layer does four things and nothing else: parse and validate input,
# translate between JSON strings and dicts, map CeremonyError to a 400, and
# manage the session cookie. All the protocol logic lives in ceremonies.py.

# Mount with:
#     from app.auth.routes import router as auth_router
#     app.include_router(auth_router)
# """

# from __future__ import annotations

# import json
# import secrets
# from dataclasses import dataclass
# from datetime import datetime, timedelta, timezone
# from typing import Annotated, Optional, Protocol, Sequence

# from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
# from pydantic import BaseModel, Field

# from app.auth.ceremonies import (
#     CeremonyError,
#     ChallengeStore,
#     Credential,
#     CredentialStore,
#     RelyingParty,
#     User,
#     begin_authentication,
#     begin_registration,
#     finish_authentication,
#     finish_registration,
# )

# SESSION_COOKIE = "session"
# SESSION_TTL = timedelta(days=14)

# router = APIRouter(prefix="/auth", tags=["auth"])


# # --------------------------------------------------------------------------
# # Sessions
# # --------------------------------------------------------------------------


# @dataclass(frozen=True)
# class Session:
#     id: str
#     user_id: str
#     expires_at: datetime


# class SessionStore(Protocol):
#     def create(self, session: Session) -> None: ...
#     def get(self, session_id: str) -> Optional[Session]: ...
#     def delete(self, session_id: str) -> None: ...
#     def delete_for_user(self, user_id: str) -> None: ...


# def _issue_session(
#     response: Response, sessions: SessionStore, user: User, *, secure: bool
# ) -> None:
#     session = Session(
#         id=secrets.token_urlsafe(32),
#         user_id=user.id,
#         expires_at=datetime.now(timezone.utc) + SESSION_TTL,
#     )
#     sessions.create(session)
#     response.set_cookie(
#         SESSION_COOKIE,
#         session.id,
#         max_age=int(SESSION_TTL.total_seconds()),
#         httponly=True,
#         # Lax keeps the cookie off cross-site POSTs, which is the CSRF defense
#         # here. Set secure=False only for http://localhost.
#         samesite="lax",
#         secure=secure,
#         path="/",
#     )


# # --------------------------------------------------------------------------
# # Dependencies
# #
# # Replace these stubs with whatever wiring you use (app.state, a container,
# # lru_cache'd factories). Keeping them as functions means tests can override
# # them with app.dependency_overrides.
# # --------------------------------------------------------------------------


# def get_rp() -> RelyingParty:
#     raise NotImplementedError("Wire this to your config.")


# def get_credentials() -> CredentialStore:
#     raise NotImplementedError("Wire this to your SQLite repository.")


# def get_challenges() -> ChallengeStore:
#     raise NotImplementedError("Wire this to your SQLite repository.")


# def get_sessions() -> SessionStore:
#     raise NotImplementedError("Wire this to your SQLite repository.")


# RP = Annotated[RelyingParty, Depends(get_rp)]
# Credentials = Annotated[CredentialStore, Depends(get_credentials)]
# Challenges = Annotated[ChallengeStore, Depends(get_challenges)]
# Sessions = Annotated[SessionStore, Depends(get_sessions)]


# def optional_user(
#     credentials: Credentials,
#     sessions: Sessions,
#     session_id: Annotated[Optional[str], Cookie(alias=SESSION_COOKIE)] = None,
# ) -> Optional[User]:
#     if not session_id:
#         return None
#     session = sessions.get(session_id)
#     if session is None or session.expires_at <= datetime.now(timezone.utc):
#         return None
#     return credentials.get_user(session.user_id)


# def current_user(user: Annotated[Optional[User], Depends(optional_user)]) -> User:
#     if user is None:
#         raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in to continue.")
#     return user


# CurrentUser = Annotated[User, Depends(current_user)]
# MaybeUser = Annotated[Optional[User], Depends(optional_user)]


# # --------------------------------------------------------------------------
# # Schemas
# # --------------------------------------------------------------------------


# class RegisterBeginRequest(BaseModel):
#     username: str = Field(min_length=1, max_length=64)
#     displayName: Optional[str] = Field(default=None, max_length=128)


# class LoginBeginRequest(BaseModel):
#     username: Optional[str] = Field(default=None, max_length=64)


# class FinishRequest(BaseModel):
#     challengeId: str = Field(min_length=1, max_length=256)
#     credential: dict




# class UserResponse(BaseModel):
#     id: str
#     username: str
#     displayName: str

#     @classmethod
#     def of(cls, user: User) -> "UserResponse":
#         return cls(id=user.id, username=user.username, displayName=user.display_name)


# class CredentialResponse(BaseModel):
#     id: str
#     nickname: Optional[str]
#     deviceType: str
#     backedUp: bool

#     @classmethod
#     def of(cls, credential: Credential) -> "CredentialResponse":
#         return cls(
#             id=credential.id,
#             nickname=credential.nickname,
#             deviceType=credential.device_type,
#             backedUp=credential.backed_up,
#         )


# # --------------------------------------------------------------------------
# # Registration
# # --------------------------------------------------------------------------



# @router.post("/register/finish", response_model=UserResponse)
# def register_finish(
#     body: FinishRequest,
#     response: Response,
#     rp: RP,
#     credentials: Credentials,
#     challenges: Challenges,
#     sessions: Sessions,
#     user: MaybeUser,
# ) -> UserResponse:
#     try:
#         credential = finish_registration(
#             rp=rp,
#             challenge_id=body.challengeId,
#             credential_json=json.dumps(body.credential),
#             credentials=credentials,
#             challenges=challenges,
#         )
#     except CeremonyError as exc:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

#     owner = credentials.get_user(credential.user_id)
#     if owner is None:  # only reachable if the user row vanished mid-ceremony
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "That account no longer exists.")

#     # Registering the first passkey signs you in; adding a second one from an
#     # existing session leaves that session alone.
#     if user is None:
#         _issue_session(response, sessions, owner, secure=rp.origin.startswith("https://"))

#     return UserResponse.of(owner)


# # --------------------------------------------------------------------------
# # Authentication
# # --------------------------------------------------------------------------


# @router.post("/login/begin", response_model=BeginResponse)
# def login_begin(
#     body: LoginBeginRequest,
#     rp: RP,
#     credentials: Credentials,
#     challenges: Challenges,
# ) -> BeginResponse:
#     challenge_id, options_json = begin_authentication(
#         rp=rp,
#         credentials=credentials,
#         challenges=challenges,
#         username=body.username,
#     )
#     return BeginResponse(challengeId=challenge_id, options=json.loads(options_json))


# @router.post("/login/finish", response_model=UserResponse)
# def login_finish(
#     body: FinishRequest,
#     response: Response,
#     rp: RP,
#     credentials: Credentials,
#     challenges: Challenges,
#     sessions: Sessions,
# ) -> UserResponse:
#     try:
#         user = finish_authentication(
#             rp=rp,
#             challenge_id=body.challengeId,
#             credential_json=json.dumps(body.credential),
#             credentials=credentials,
#             challenges=challenges,
#         )
#     except CeremonyError as exc:
#         # Deliberately uniform: never distinguish "no such credential" from
#         # "signature didn't verify" in the response body.
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

#     _issue_session(response, sessions, user, secure=rp.origin.startswith("https://"))
#     return UserResponse.of(user)


# # --------------------------------------------------------------------------
# # Session and credential management
# # --------------------------------------------------------------------------


# @router.get("/me", response_model=UserResponse)
# def me(user: CurrentUser) -> UserResponse:
#     return UserResponse.of(user)


# @router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
# def logout(
#     response: Response,
#     sessions: Sessions,
#     session_id: Annotated[Optional[str], Cookie(alias=SESSION_COOKIE)] = None,
# ) -> None:
#     if session_id:
#         sessions.delete(session_id)
#     response.delete_cookie(SESSION_COOKIE, path="/")


# @router.get("/credentials", response_model=list[CredentialResponse])
# def list_credentials(user: CurrentUser, credentials: Credentials) -> list[CredentialResponse]:
#     return [CredentialResponse.of(c) for c in credentials.credentials_for_user(user.id)]


# @router.delete("/credentials/{credential_id}", status_code=status.HTTP_204_NO_CONTENT)
# def delete_credential(
#     credential_id: str,
#     user: CurrentUser,
#     credentials: Credentials,
# ) -> None:
#     owned: Sequence[Credential] = credentials.credentials_for_user(user.id)
#     target = next((c for c in owned if c.id == credential_id), None)
#     if target is None:
#         raise HTTPException(status.HTTP_404_NOT_FOUND, "No such passkey.")

#     # Removing the last passkey locks the account out permanently, since
#     # there's nothing else to authenticate with. Add your recovery path
#     # before relaxing this.
#     if len(owned) == 1:
#         raise HTTPException(
#             status.HTTP_409_CONFLICT,
#             "This is your only passkey. Add another before removing this one.",
#         )

#     credentials.delete_credential(credential_id)
