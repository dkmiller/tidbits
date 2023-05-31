import argparse
import asyncio
import json
import os

from aiohttp import ClientSession
from unidiff import PatchSet


from gidgethub import aiohttp as gh_aiohttp


PLACEHOLDER_COMMENT = "No comments yet."
LINE_LENGTH = 100
PROMPT_PREFIX = "You are my code review assistant. I will give you a git diff below. Review the diff and suggest any improvements or changes.\n\n"


def handle_row(fname, prompt, completion):
    row = {"prompt": prompt, "completion": completion}
    if completion != PLACEHOLDER_COMMENT:
        print("=" * LINE_LENGTH)
        print(prompt)
        print("-" * LINE_LENGTH)
        print(completion)
        print("=" * LINE_LENGTH)
    else:
        print(f"===== Writing placeholder comment for prompt =====")
    # print(comment['body'])
    # print("=" * LINE_LENGTH)

    # https://stackoverflow.com/a/57740064
    with open(fname, "a+") as f:
        json.dump(row, f)
        f.write(os.linesep)


async def main(args):
    async with ClientSession() as session:
        gh = gh_aiohttp.GitHubAPI(session, "dkmiller", oauth_token=args.token)
        # rows = []
        # This handles pagination.
        async for pull in gh.getiter(f"/repos/{args.repo}/pulls?state=all"):
            # row = {}
            print(pull["url"])

            # https://github.com/orgs/community/discussions/24460#discussioncomment-3244157
            diff = await gh.getitem(f"/repos/{args.repo}/pulls/{pull['number']}", accept="application/vnd.github.v3.diff")
            if not diff:
                continue
            # row["prompt"] = f"You are my code review assistant. I will give you a git diff below. Review the diff and suggest any improvements or changes.\n\n{diff}"

            # completion = ""




            # x.append(pull)
            # print(pull)
            # print("=" * 200)
            # print(f"/repos/{args.repo}/pulls/{pull['number']}/comments")
            # https://stackoverflow.com/a/62585232
            comments = [c async for c in gh.getiter(f"/repos/{args.repo}/pulls/{pull['number']}/comments")]
            patch_set = PatchSet(diff)

            for patch in patch_set:
                addressed = False
                for comment in comments:
                    if comment["path"] in patch.source_file or comment["path"] in patch.target_file:
                        handle_row(args.output, PROMPT_PREFIX + str(patch), comment["body"])
                        # rows.append({
                        #     "prompt": PROMPT_PREFIX + str(patch),
                        #     "completion": comment["body"],
                        # })
                        # row["prompt"] = PROMPT_PREFIX + str(diff)



                        # print("=" * LINE_LENGTH)
                        # print(row)
                        # print("=" * LINE_LENGTH)
                        # print(str(patch))
                        # print("-" * LINE_LENGTH)
                        # print(comment['body'])
                        # print("=" * LINE_LENGTH)
                        addressed = True
                if not addressed:
                    handle_row(args.output, PROMPT_PREFIX + str(patch), PLACEHOLDER_COMMENT)

                        # rows.append({
                        #     "prompt": PROMPT_PREFIX + str(patch),
                        #     "completion":  PLACEHOLDER_COMMENT
                        # })
                        # print("=" * LINE_LENGTH)
                        # print(rows[-1])
                        # print("=" * LINE_LENGTH)




            # async for comment in gh.getiter(f"/repos/{args.repo}/pulls/{pull['number']}/comments"):
            #     completion += f"\n{comment['body']}\n"
                # # https://api.github.com/repos/airbnb/lottie-web/pulls/2971/comments
                # print(f"========== comment {comment['id']} >>>>>>>>>>")
                # print(comment["body"])
                # print("-" * 200)
                # print(comment["diff_hunk"])
                # print(f"========== comment {comment['id']} <<<<<<<<<<")
                # print(gh.rate_limit.remaining)
                # if gf.rate_

    #         row["completion"] = completion or "No comments yet."
    #         print(f"========== pull {pull['id']} >>>>>>>>>>")
    #         print(row["prompt"])
    #         print("-" * 100)
    #         print(row["completion"])
    #         print(f"========== pull {pull['id']} <<<<<<<<<<")
    #         rows.append(row)

    # import json, os
    # # https://stackoverflow.com/a/57740064
    # with open(args.output, "w+") as f:
    #     for row in rows:
    #         json.dump(row, f)
    #         f.write(os.linesep)

            # import sys
            # sys.exit(0)
        # print(len(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="airbnb/lottie-web")
    parser.add_argument("--token", required=True)
    parser.add_argument("--output", default="output.jsonl")
    args = parser.parse_args()
    asyncio.run(main(args))
