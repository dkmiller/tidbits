import argparse
import asyncio
from dataclasses import dataclass

from aiohttp import ClientSession


@dataclass
class GithubClient:
    """
    TODO: why not https://gidgethub.readthedocs.io/
    """
    repo: str
    session: ClientSession

    async def get(self, path: str):
        url = f"https://api.github.com/repos/{self.repo}/{path}"
        async with self.session.get(url) as response:
            return await response.json()

# TODO: this didn't work:
# gidgethub.BadRequest: Not Found
from gidgethub import aiohttp as gh_aiohttp


async def main(args):
    async with ClientSession() as session:
        gh = gh_aiohttp.GitHubAPI(session, "dkmiller")
        # x = await gh.getitem("/repos/brettcannon/gidgethub/labels/bug")
        # print(x)
        x = []
        # This handles pagination.
        async for pull in  gh.getiter("/repos/airbnb/lottie-web/pulls?state=all"):
            print(pull["url"])
            x.append(pull)
        print(len(x))
        # await gh.getitem("/repos/brettcannon/gidgethub/labels/bug")

        # client = GithubClient(args.repo, session)
        # pulls = await client.get("pulls?per_page=100&state=closed")
        # print(len(pulls))
        # print(pulls[0])
        # print('GH requests remaining:', gh.rate_limit.remaining)
        # try:
        #     async for p in gh.getiter("/pulls"):
        #         print(p)
        # except Exception as e:
        #     print(e)
        # pulls = await gh.getiter("/pulls/2976")
        # print(pulls)

    # TODO: use https://gidgethub.readthedocs.io/en/latest/ instead? 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="airbnb/lottie-web")
    args = parser.parse_args()
    asyncio.run(main(args))
 