from webauthn import generate_registration_options, options_to_json
from webauthn.helpers import base64url_to_bytes, bytes_to_base64url
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    AuthenticatorTransport,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

from tidbits.models import RelyingParty, User
from tidbits.users import ChallengeStore, UserStore

CEREMONY_TIMEOUT_MS = 60_000


async def begin_registration(
    *,
    rp: RelyingParty,
    user: User,
    user_store: UserStore,
    challenges: ChallengeStore,
    require_resident_key: bool = True,
) -> tuple[str, str]:
    """
    Returns (challenge_id, options_json).

    Hand challenge_id back to the client as an opaque token — a short-lived
    cookie or a field in the JSON body. Do not let the client send the
    challenge itself; it must come from the server's own record.
    """
    existing = await user_store.credentials_for_user(user.id)

    options = generate_registration_options(
        rp_id=rp.id,
        rp_name=rp.name,
        # The user handle is written into the authenticator and is visible to
        # anyone who inspects the device. Keep it opaque.
        user_id=str(user.id).encode("utf-8"),
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

    challenge_id = await challenges.store(
        options.challenge, user_id=user.id, kind="registration"
    )
    return challenge_id, options_to_json(options)
