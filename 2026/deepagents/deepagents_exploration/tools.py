import asyncio
from typing import Any

from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport


# https://docs.github.com/en/graphql/guides/using-graphql-clients
# https://gql.readthedocs.io/en/latest/usage/async_usage.html#async-usage


async def github_graphql(query: str, variables: dict[str, Any] | None = None):
    pass


async def main():

    # Select your transport with a defined url endpoint
    import os
    token = os.getenv("GITHUB_PAT")
    transport = AIOHTTPTransport(url="https://api.github.com/graphql", headers={"Authorization": f"Bearer {token}"})

    # Create a GraphQL client using the defined transport
    client = Client(transport=transport)

    # Provide a GraphQL query
    query = gql(
        """
        query {
  viewer {
    login
  }
}
    """
    )

    # Using `async with` on the client will start a connection on the transport
    # and provide a `session` variable to execute queries on this connection
    async with client as session:

        # Execute the query
        result = await session.execute(query)
        print(result)


asyncio.run(main())
