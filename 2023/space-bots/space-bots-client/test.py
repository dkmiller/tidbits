from space_bots_client import AuthenticatedClient
from space_bots_client.api.fleets import get_fleets_my
import os

client = AuthenticatedClient(
    "https://space-bots.longwelwind.net/v1/", os.environ["SPACEBOTS_TOKEN"]
)

print(client)


print(get_fleets_my.sync_detailed(client=client))
