import argparse
from gfycat.client import GfycatClient
from lxml import html
import praw
import requests
from urllib.parse import urlparse


class _SafeOperation:
    def __init__(self, indent: str):
        self.indent = indent

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            print(f"{self.indent}Failed: {exc_value}")

        # https://stackoverflow.com/a/34113126
        return not isinstance(exc_value, KeyboardInterrupt)


def safe_operation(indent: str = "\t"):
    return _SafeOperation(indent)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gfycat_id", default="2_sdSGQD")
    parser.add_argument("--output", default=None)
    parser.add_argument("--reddit_id", default="XnkMHEYUujv1wA7EkmToWg")
    parser.add_argument("--gfycat_secret", required=True, type=str)
    parser.add_argument("--reddit_secret", required=True, type=str)
    parser.add_argument("--limit", default=10, type=int)
    parser.add_argument("--subreddit", required=True, type=str)
    parser.add_argument("--user_agent", default="praw.ml")
    return parser


def get_gfycat_client(args) -> GfycatClient:
    rv = GfycatClient(args.gfycat_id, args.gfycat_secret)
    return rv


def get_reddit_client(args) -> praw.Reddit:
    rv = praw.Reddit(
        client_id=args.reddit_id,
        client_secret=args.reddit_secret,
        user_agent=args.user_agent,
    )
    return rv


def extract(element, attribute: str, extension: str) -> list:
    rv = []
    with safe_operation():
        attr_value = element.attrib.get(attribute, None)
        if attr_value and attr_value.endswith(extension):
            rv.append(attr_value)

    with safe_operation():
        children = element.getchildren()
        for child in children:
            child_extraction = extract(child, attribute, extension)
            rv.extend(child_extraction)

    return rv


def get_official_gfycat_urls(gfycat, url: str) -> list:
    rv = []
    parsed_url = urlparse(url)
    if parsed_url.netloc == "gfycat.com":
        name = parsed_url.path.split("/")[-1]
        with safe_operation():
            gfycat_url = gfycat.query_gfy(name)["gfyItem"]["mp4Url"]
            rv.append(gfycat_url)
    return rv


def get_adhoc_gfycat_urls(reddit, url: str) -> list:
    rv = []

    with safe_operation():
        r = requests.get(url)
        tree = html.fromstring(r.text)
        gif_urls = extract(tree, "content", ".mp4")
        rv.extend(gif_urls)

    return list(set(rv))


def main(args):
    reddit = get_reddit_client(args)
    gfycat = get_gfycat_client(args)
    posts = reddit.subreddit(args.subreddit).top("all", limit=args.limit)

    for post in posts:
        message = f"{post.title}\n    {post.url}\n"
        print(message)

        official_urls = get_official_gfycat_urls(gfycat, post.url)
        adhoc_urls = get_adhoc_gfycat_urls(reddit, post.url)

        for url in official_urls + adhoc_urls:
            append = f"    {url}\n"
            print(append)
            message += append

        if args.output:
            with open(args.output, "a") as f:
                f.write(message)


if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)
