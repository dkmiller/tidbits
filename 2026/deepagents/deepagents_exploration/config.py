import os
from functools import lru_cache

from dotenv import dotenv_values, load_dotenv, set_key
from onepassword.client import Client, DesktopAuth


@lru_cache(maxsize=1)
def onepassword_client():
    return Client.authenticate(
        auth=DesktopAuth(account_name="Miller Family"),
        integration_name="tidbits-deepagents",
        integration_version="v0.0.1",
    )


async def initialize_environment():
    values = dotenv_values("env.template")
    load_dotenv()

    for key, value in values.items():
        if key in os.environ or not value:
            continue
        if value.startswith("op://"):
            client = await onepassword_client()
            resolved_value = await client.secrets.resolve(value)
            os.environ[key] = resolved_value
            set_key(".env", key, resolved_value)
        else:
            os.environ[key] = value
            set_key(".env", key, value)
