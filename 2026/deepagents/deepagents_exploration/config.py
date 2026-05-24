import os

from dotenv import dotenv_values
from onepassword.client import Client, DesktopAuth


async def initialize_environment():
    values = dotenv_values()

    client = await Client.authenticate(
        auth=DesktopAuth(account_name="Miller Family"),
        integration_name="tidbits-deepagents",
        integration_version="v0.0.1",
    )

    for key, value in values.items():
        if not value:
            continue
        if value.startswith("op://"):
            resolved_value = await client.secrets.resolve(value)
            os.environ[key] = resolved_value
        else:
            os.environ[key] = value
