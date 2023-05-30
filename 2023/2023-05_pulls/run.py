import argparse
import asyncio

from aiohttp import ClientSession


from gidgethub import aiohttp as gh_aiohttp


async def main(args):
    async with ClientSession() as session:
        gh = gh_aiohttp.GitHubAPI(session, "dkmiller")
        x = []
        # This handles pagination.
        async for pull in gh.getiter(f"/repos/{args.repo}/pulls?state=all"):
            print(pull["url"])
            x.append(pull)
        print(len(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="airbnb/lottie-web")
    args = parser.parse_args()
    asyncio.run(main(args))
