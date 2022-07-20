from aiohttp import ClientSession
import argparse
import asyncio
from azure.identity import DefaultAzureCredential
from typing import List
from uuid import uuid4


class AsyncAzureClient:
    def __init__(self, credential, session: ClientSession):
        token = credential.get_token("https://management.azure.com/.default")
        self.headers = {"Authorization": f"Bearer {token.token}"}
        self.session = session
        graph_token = credential.get_token("https://graph.microsoft.com/.default")
        self.graph_headers = {"Authorization": f"Bearer {graph_token.token}"}

    async def get_subscriptions(self) -> List[str]:
        """
        https://docs.microsoft.com/en-us/rest/api/resources/subscriptions/list
        """
        url = "https://management.azure.com/subscriptions?api-version=2020-01-01"
        async with self.session.get(url, headers=self.headers) as r:
            json = await r.json()
            rv = [x["subscriptionId"] for x in json["value"]]
            return rv

    async def assign_role(self, scope, role, principal):
        """
        https://docs.microsoft.com/en-us/azure/role-based-access-control/role-assignments-rest
        """
        role_assignment_id = str(uuid4())
        scope_provider = f"/{scope}/providers/Microsoft.Authorization/"
        url = f"https://management.azure.com{scope_provider}roleAssignments/{role_assignment_id}?api-version=2015-07-01"
        body = {
            "properties": {
                "roleDefinitionId": f"{scope_provider}roleDefinitions/{role}",
                "principalId": principal,
            }
        }
        async with self.session.put(url, json=body, headers=self.headers) as r:
            return r.status

    async def get_user_oid(self, name: str) -> str:
        """
        https://docs.microsoft.com/en-us/graph/api/user-list
        """
        url = (
            f"https://graph.microsoft.com/v1.0/users?$filter=(displayName eq '{name}')"
        )
        async with self.session.get(url, headers=self.graph_headers) as r:
            json = await r.json()
            rv = json["value"][0]["id"]
            return rv


async def main_async(new_admin: str):
    credential = DefaultAzureCredential()
    async with ClientSession() as session:
        client = AsyncAzureClient(credential, session)
        oid = await client.get_user_oid(new_admin)
        subscriptions = await client.get_subscriptions()
        print(f"Adding {oid} to {len(subscriptions)} subscriptions")

        statuses = await asyncio.gather(
            *[
                # https://docs.microsoft.com/en-us/azure/role-based-access-control/built-in-roles
                client.assign_role(
                    f"subscriptions/{s}", "8e3af657-a8ff-443c-a75c-2fe8c4bcb635", oid
                )
                for s in subscriptions
            ]
        )

        for i, subscription in enumerate(subscriptions):
            print(f"{subscription}: {statuses[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_admin", required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args.new_admin))
