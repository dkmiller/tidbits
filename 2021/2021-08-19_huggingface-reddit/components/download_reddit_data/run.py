from argparse_dataclass import ArgumentParser
from azureml.core import Run
from azureml.core.run import _SubmittedRun
from dataclasses import dataclass
from itertools import groupby, count
import json
import jsonpickle
import logging
from multiprocessing import Pool
import os
from pathlib import Path
from praw import Reddit
import re
import sys


log = logging.getLogger(__name__)


def remove_private_keys(o):
    if not isinstance(o, dict):
        return
    public_keys = [k for k in o.keys() if k.startswith("_")]
    for k in public_keys:
        del o[k]
    for v in o.values():
        remove_private_keys(v)


def serialize(post) -> str:
    """
    Clean serialization of "raw" Praw submission object. Necessary because
    existing serialization libraries persist lots of private Praw information,
    and the Praw team refuses at support access to the "raw" HTTP response.

    WARNING: this is memory intensive, since it (de)serializes the object
    multiple times.

    https://github.com/praw-dev/praw/issues/830
    """
    # https://stackoverflow.com/a/8614096
    raw_json = jsonpickle.encode(post, unpicklable=False)
    raw_object = json.loads(raw_json)
    remove_private_keys(raw_object)
    return json.dumps(raw_object)


@dataclass
class Args:
    output_directory: str
    client_id: str
    client_secret: str
    subreddits: str
    top_mode: str
    logging_level: str = "INFO"
    post_limit: int = 1000
    posts_per_file: int = 10


def resolve_secret(s: str) -> str:
    if m := re.match(r"secret://(.+)", s):
        secret_name = m.group(1)
        log.info(f"Resolving secret '{secret_name}' from Azure ML")
        # https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-secrets-in-runs
        run = Run.get_context()
        if not isinstance(run, _SubmittedRun):
            raise RuntimeError("Secret resolution only works inside Azure ML")
        s = run.get_secret(secret_name)
    return s


def download_subreddit(subreddit: str, reddit: Reddit, args: Args):
    log.info(
        f"Downloading the top {args.post_limit} posts from '{subreddit}' by {args.top_mode}"
    )
    posts = reddit.subreddit(subreddit).top(args.top_mode, limit=args.post_limit)

    # https://stackoverflow.com/a/40063403
    c = count()
    post_chunks = groupby(posts, lambda _: next(c) // args.posts_per_file)
    for index, chunk in post_chunks:
        file_name = f"top_{args.posts_per_file}__{index}.json"
        file_path = Path(args.output_directory) / subreddit / file_name
        log.info(f"Writing <={args.posts_per_file} posts to {file_path}")
        # https://stackoverflow.com/a/62348146
        file_path.parent.mkdir(parents=True, exist_ok=True)
        for post in chunk:
            log.debug(f"Writing {subreddit}/{post.id}")
            post_raw = serialize(post)
            # https://stackoverflow.com/a/57345569
            with file_path.open("a") as fp:
                fp.write(post_raw)
                fp.write(os.linesep)


def main(args: Args):
    log.info(f"Running using arguments {args}")
    client_secret = resolve_secret(args.client_secret)
    reddit = Reddit(
        client_id=args.client_id, client_secret=client_secret, user_agent="praw.ml"
    )

    # https://stackoverflow.com/a/30933281
    subreddits = list(filter(None, re.split(r"\s|,|;", args.subreddits)))
    pool = Pool()
    # https://stackoverflow.com/a/20354129
    log.info(
        f"Downloading {len(subreddits)} sub-reddits using {pool._processes} processes."
    )

    # https://replit.com/@allasamhita/Pool-apply
    for subreddit in subreddits:
        pool.apply(download_subreddit, args=[subreddit, reddit, args])
    pool.close()
    pool.join()
    # posts = reddit.subreddit(subreddit).top(args.top_mode, limit=args.post_limit)

    # # https://stackoverflow.com/a/40063403
    # c = count()
    # post_chunks = groupby(posts, lambda _: next(c) // args.posts_per_file)
    # for index, chunk in post_chunks:
    #     file_name = f"top_{args.posts_per_file}__{index}.json"
    #     file_path = Path(args.output_directory) / subreddit / file_name
    #     log.info(f"Writing <={args.posts_per_file} posts to {file_path}")
    #     # https://stackoverflow.com/a/62348146
    #     file_path.parent.mkdir(parents=True, exist_ok=True)
    #     for post in chunk:
    #         log.debug(f"Writing {subreddit}/{post.id}")
    #         post_raw = serialize(post)
    #         # https://stackoverflow.com/a/57345569
    #         with file_path.open("a") as fp:
    #             fp.write(post_raw)
    #             fp.write(os.linesep)


if __name__ == "__main__":
    print(f"Unparsed args: {sys.argv}")

    parser = ArgumentParser(Args)
    args = parser.parse_args()
    logging.basicConfig(level=args.logging_level)
    main(args)
