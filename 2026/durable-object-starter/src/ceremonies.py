# TODO DELETE THIS FILE!!! It is being merged into app.py, models.py and users.py.

"""
WebAuthn registration and authentication ceremonies.

Deliberately free of FastAPI: everything here takes plain arguments and
returns plain data, so it can be tested without a browser or an HTTP client.
Persistence is reached through the two Protocols below — implement them
against SQLite (or anything else) in your repository layer.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol, Sequence

from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers import base64url_to_bytes, bytes_to_base64url
from webauthn.helpers.exceptions import (
    InvalidAuthenticationResponse,
    InvalidRegistrationResponse,
)
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    AuthenticatorTransport,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

CHALLENGE_TTL = timedelta(seconds=60)
CEREMONY_TIMEOUT_MS = 60_000


# --------------------------------------------------------------------------
# Domain types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class User:
    id: str  # opaque user handle, NOT the username
    username: str
    display_name: str


@dataclass(frozen=True)
class Credential:
    id: str  # base64url credential ID
    user_id: str
    public_key: bytes  # COSE key
    sign_count: int
    transports: list[str]
    aaguid: Optional[str] = None
    backed_up: bool = False
    device_type: str = "single_device"
    nickname: Optional[str] = None


@dataclass(frozen=True)
class PendingChallenge:
    id: str
    challenge: bytes
    user_id: Optional[str]
    kind: str  # "registration" | "authentication"
    expires_at: datetime


class CredentialStore(Protocol):
    def get_user(self, user_id: str) -> Optional[User]: ...
    def get_user_by_username(self, username: str) -> Optional[User]: ...
    def credentials_for_user(self, user_id: str) -> Sequence[Credential]: ...
    def get_credential(self, credential_id: str) -> Optional[Credential]: ...
    def add_credential(self, credential: Credential) -> None: ...
    def update_sign_count(self, credential_id: str, sign_count: int) -> None: ...


class ChallengeStore(Protocol):
    def put(self, challenge: PendingChallenge) -> None: ...
    def take(self, challenge_id: str) -> Optional[PendingChallenge]:
        """Fetch and delete in one step. Single use, always."""
        ...


class CeremonyError(Exception):
    """Anything that should surface to the client as a 400."""


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RelyingParty:
    id: str  # "example.com" — registrable domain, no scheme, no port
    name: str  # "Acme"
    origin: str  # "https://example.com" — full origin the browser sees

    # For local development this is typically:
    #   RelyingParty(id="localhost", name="Acme (dev)", origin="http://localhost:8000")


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def begin_registration(
    *,
    rp: RelyingParty,
    user: User,
    credentials: CredentialStore,
    challenges: ChallengeStore,
    require_resident_key: bool = True,
) -> tuple[str, str]:
    """
    Returns (challenge_id, options_json).

    Hand challenge_id back to the client as an opaque token — a short-lived
    cookie or a field in the JSON body. Do not let the client send the
    challenge itself; it must come from the server's own record.
    """
    existing = credentials.credentials_for_user(user.id)

    options = generate_registration_options(
        rp_id=rp.id,
        rp_name=rp.name,
        # The user handle is written into the authenticator and is visible to
        # anyone who inspects the device. Keep it opaque.
        user_id=user.id.encode("utf-8"),
        user_name=user.username,
        user_display_name=user.display_name,
        # Stops an authenticator that already holds a credential for this
        # account from silently creating a second one.
        exclude_credentials=[
            PublicKeyCredentialDescriptor(
                id=base64url_to_bytes(c.id),
                transports=[AuthenticatorTransport(t) for t in c.transports] or None,
            )
            for c in existing
        ],
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=(
                ResidentKeyRequirement.REQUIRED
                if require_resident_key
                else ResidentKeyRequirement.DISCOURAGED
            ),
            user_verification=UserVerificationRequirement.PREFERRED,
        ),
        timeout=CEREMONY_TIMEOUT_MS,
    )

    challenge_id = _store_challenge(
        challenges, options.challenge, user_id=user.id, kind="registration"
    )
    return challenge_id, options_to_json(options)


def finish_registration(
    *,
    rp: RelyingParty,
    challenge_id: str,
    credential_json: str,
    credentials: CredentialStore,
    challenges: ChallengeStore,
    nickname: Optional[str] = None,
) -> Credential:
    pending = _take_valid_challenge(challenges, challenge_id, expected_kind="registration")
    if pending.user_id is None:
        raise CeremonyError("Registration challenge is missing its user.")

    try:
        verified = verify_registration_response(
            credential=credential_json,
            expected_challenge=pending.challenge,
            expected_origin=rp.origin,
            expected_rp_id=rp.id,
            require_user_verification=False,
        )
    except InvalidRegistrationResponse as exc:
        raise CeremonyError(f"Could not verify the new credential: {exc}") from exc

    credential_id = bytes_to_base64url(verified.credential_id)
    if credentials.get_credential(credential_id) is not None:
        raise CeremonyError("That credential is already registered.")

    # Transports come from the client response, not from verification output.
    transports = json.loads(credential_json).get("response", {}).get("transports", [])

    credential = Credential(
        id=credential_id,
        user_id=pending.user_id,
        public_key=verified.credential_public_key,
        sign_count=verified.sign_count,
        transports=[t for t in transports if isinstance(t, str)],
        aaguid=verified.aaguid,
        backed_up=verified.credential_backed_up,
        device_type=verified.credential_device_type.value,
        nickname=nickname,
    )
    credentials.add_credential(credential)
    return credential


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


def begin_authentication(
    *,
    rp: RelyingParty,
    credentials: CredentialStore,
    challenges: ChallengeStore,
    username: Optional[str] = None,
) -> tuple[str, str]:
    """
    Pass a username to scope the ceremony to that account's credentials.
    Omit it for the usernameless flow: allow_credentials stays empty and the
    browser offers whatever discoverable credentials it holds for this RP.

    Note the deliberate lack of an error when the username is unknown — an
    empty allow list is indistinguishable from the usernameless flow, so an
    attacker learns nothing about which accounts exist.
    """
    allow: list[PublicKeyCredentialDescriptor] = []
    user_id: Optional[str] = None

    if username:
        user = credentials.get_user_by_username(username)
        if user is not None:
            user_id = user.id
            allow = [
                PublicKeyCredentialDescriptor(
                    id=base64url_to_bytes(c.id),
                    transports=[AuthenticatorTransport(t) for t in c.transports] or None,
                )
                for c in credentials.credentials_for_user(user.id)
            ]

    options = generate_authentication_options(
        rp_id=rp.id,
        allow_credentials=allow,
        user_verification=UserVerificationRequirement.PREFERRED,
        timeout=CEREMONY_TIMEOUT_MS,
    )

    challenge_id = _store_challenge(
        challenges, options.challenge, user_id=user_id, kind="authentication"
    )
    return challenge_id, options_to_json(options)


def finish_authentication(
    *,
    rp: RelyingParty,
    challenge_id: str,
    credential_json: str,
    credentials: CredentialStore,
    challenges: ChallengeStore,
) -> User:
    pending = _take_valid_challenge(challenges, challenge_id, expected_kind="authentication")

    parsed = json.loads(credential_json)
    credential_id = parsed.get("id")
    if not isinstance(credential_id, str):
        raise CeremonyError("The response is missing a credential ID.")

    stored = credentials.get_credential(credential_id)
    if stored is None:
        raise CeremonyError("Unrecognized credential.")

    # If the ceremony was scoped to an account, the credential must belong to
    # it. Without this check a user could start a login as A and finish as B.
    if pending.user_id is not None and stored.user_id != pending.user_id:
        raise CeremonyError("Unrecognized credential.")

    try:
        verified = verify_authentication_response(
            credential=credential_json,
            expected_challenge=pending.challenge,
            expected_origin=rp.origin,
            expected_rp_id=rp.id,
            credential_public_key=stored.public_key,
            credential_current_sign_count=stored.sign_count,
            require_user_verification=False,
        )
    except InvalidAuthenticationResponse as exc:
        raise CeremonyError(f"Could not verify that credential: {exc}") from exc

    # Plenty of authenticators always report 0. Only a non-zero counter that
    # fails to advance is meaningful, and even then it's a signal to log and
    # investigate rather than a reason to lock someone out.
    if stored.sign_count > 0 and verified.new_sign_count <= stored.sign_count:
        _log_possible_clone(stored, verified.new_sign_count)

    credentials.update_sign_count(stored.id, verified.new_sign_count)

    user = credentials.get_user(stored.user_id)
    if user is None:
        raise CeremonyError("Unrecognized credential.")
    return user


# --------------------------------------------------------------------------
# Challenge plumbing
# --------------------------------------------------------------------------


def _store_challenge(
    challenges: ChallengeStore, challenge: bytes, *, user_id: Optional[str], kind: str
) -> str:
    challenge_id = secrets.token_urlsafe(32)
    challenges.put(
        PendingChallenge(
            id=challenge_id,
            challenge=challenge,
            user_id=user_id,
            kind=kind,
            expires_at=datetime.now(timezone.utc) + CHALLENGE_TTL,
        )
    )
    return challenge_id


def _take_valid_challenge(
    challenges: ChallengeStore, challenge_id: str, *, expected_kind: str
) -> PendingChallenge:
    pending = challenges.take(challenge_id)
    if pending is None:
        raise CeremonyError("That sign-in attempt is no longer valid. Start again.")
    if pending.kind != expected_kind:
        raise CeremonyError("That sign-in attempt is no longer valid. Start again.")
    if pending.expires_at <= datetime.now(timezone.utc):
        raise CeremonyError("That sign-in attempt expired. Start again.")
    return pending


def _log_possible_clone(credential: Credential, new_sign_count: int) -> None:
    import logging

    logging.getLogger(__name__).warning(
        "Signature counter did not advance for credential %s (stored=%d, received=%d). "
        "Possible cloned authenticator.",
        credential.id,
        credential.sign_count,
        new_sign_count,
    )
