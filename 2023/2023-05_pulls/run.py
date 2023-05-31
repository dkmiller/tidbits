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
            # print(pull)
            # print("=" * 200)
            # print(f"/repos/{args.repo}/pulls/{pull['number']}/comments")
            async for comment in gh.getiter(f"/repos/{args.repo}/pulls/{pull['number']}/comments"):
                # https://api.github.com/repos/airbnb/lottie-web/pulls/2971/comments
                print(f"========== comment {comment['id']} >>>>>>>>>>")
                print(comment["body"])
                print("-" * 200)
                print(comment["diff_hunk"])
                print(f"========== comment {comment['id']} <<<<<<<<<<")
                print(gh.rate_limit.remaining)
                # if gf.rate_

            # import sys
            # sys.exit(0)
        print(len(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="airbnb/lottie-web")
    args = parser.parse_args()
    asyncio.run(main(args))
