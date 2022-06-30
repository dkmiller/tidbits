from aiohttp import ClientSession, ContentTypeError
import argparse
import asyncio
from azure.identity import DefaultAzureCredential
from typing import List


class AsyncAzdoClient:
    def __init__(self, credential, session: ClientSession):
        token = credential.get_token("499b84ac-1321-427f-aa17-267ca6975798/.default")
        self.headers = {"Authorization": f"Bearer {token.token}"}
        self.session = session

    async def get_public_alias(self) -> str:
        # https://stackoverflow.com/a/67871296
        url = "https://app.vssps.visualstudio.com/_apis/profile/profiles/me?api-version=6.0"
        async with self.session.get(url, headers=self.headers) as r:
            json = await r.json()
            return json["publicAlias"]

    async def get_organizations(self, public_alias: str) -> List[str]:
        url = f"https://app.vssps.visualstudio.com/_apis/accounts?memberId={public_alias}&api-version=6.0"
        async with self.session.get(url, headers=self.headers) as r:
            json = await r.json()
            rv = [x["accountName"] for x in json["value"]]
            return rv

    async def get_projects(self, organization: str) -> List[str]:
        """
        Returns project **IDs**.
        """
        url = f"https://dev.azure.com/{organization}/_apis/projects?api-version=6.0"
        async with self.session.get(url, headers=self.headers) as r:
            try:
                json = await r.json()
            # Happens when AzDO instance is deactivated:
            # https://aka.ms/adofootprintreduction
            except ContentTypeError:
                return []
            rv = [x["id"] for x in json["value"]]
            return rv

    async def service_connections(
        self, organization: str, project: str, public_alias: str
    ) -> List[str]:
        """
        Returns service connection / endpoint **IDs**.
        """
        url = f"https://dev.azure.com/{organization}/{project}/_apis/serviceendpoint/endpoints?api-version=6.0-preview.4"
        async with self.session.get(url, headers=self.headers) as r:
            try:
                json = await r.json()
            # Happens when AzDO instance is deactivated:
            # https://aka.ms/adofootprintreduction
            except ContentTypeError:
                return []

            rv = [
                x["id"] for x in json["value"] if x["createdBy"]["id"] == public_alias
            ]
            return rv

    async def add_admin_to_service_connection(
        self,
        organization: str,
        project: str,
        service_connection: str,
        public_alias: str,
    ):
        # https://stackoverflow.com/a/59288659
        url = f"https://dev.azure.com/{organization}/_apis/securityroles/scopes/distributedtask.serviceendpointrole/roleassignments/resources/{project}_{service_connection}?api-version=5.1-preview"
        body = [{"roleName": "Administrator", "userId": public_alias}]
        async with self.session.put(url, headers=self.headers, json=body) as r:
            return r.status


async def main_async(new_admin: str):
    credential = DefaultAzureCredential()
    async with ClientSession() as session:
        client = AsyncAzdoClient(credential, session)
        public_alias = await client.get_public_alias()
        organizations = await client.get_organizations(public_alias)
        projects = await asyncio.gather(*map(client.get_projects, organizations))

        flattened_projects = [
            (o, p) for i, o in enumerate(organizations) for p in projects[i]
        ]
        service_connections = await asyncio.gather(
            *[
                client.service_connections(o, p, public_alias)
                for o, p in flattened_projects
            ]
        )

        flattened_connections = [
            (o, p, c)
            for i, (o, p) in enumerate(flattened_projects)
            for c in service_connections[i]
        ]

        statuses = await asyncio.gather(
            *[
                client.add_admin_to_service_connection(o, p, c, new_admin)
                for (o, p, c) in flattened_connections
            ]
        )

        print(statuses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_admin", required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args.new_admin))
