import argparse
from gfycat.client import GfycatClient
from lxml import html
import praw
import requests
from urllib.parse import urlparse


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
    try:
        attr_value = element.attrib.get(attribute, None)
        if attr_value and attr_value.endswith(extension):
            rv.append(attr_value)
    except Exception as e:
        print(f"\t{e}")

    try:
        children = element.getchildren()
        for child in children:
            child_extraction = extract(child, attribute, extension)
            rv.extend(child_extraction)
    except BaseException as e:
        print(f"\t{e}")

    return rv


def main(args):
    reddit = get_reddit_client(args)
    gfycat = get_gfycat_client(args)
    posts = reddit.subreddit(args.subreddit).top("all", limit=args.limit)

    for post in posts:
        message = f"{post.title}\n    {post.url}\n"
        print(message)
        if args.output:
            with open(args.output, "a") as f:
                f.write(message)
        parsed_url = urlparse(post.url)

        if parsed_url.netloc == "gfycat.com":
            name = parsed_url.path.split("/")[-1]
            try:
                gfycat_url = gfycat.query_gfy(name)["gfyItem"]["mp4Url"]
                print(f"\tGfycat URL: {gfycat_url}")
            except BaseException as e:
                print(f"\tFailed: {e}")

        try:
            r = requests.get(post.url)
            tree = html.fromstring(r.text)
            # print(tree)
            # next_links = [
            #     n
            #     for n in [
            #         c.attrib.get("content", None)
            #         for c in tree.getchildren()[0].getchildren()
            #     ]
            #     if n and n.endswith(".mp4")
            # ]
            # if next_links:
            #     print(next_links)
            new_next_links = extract(tree, "content", ".mp4")
            if new_next_links:
                print(f"\tAd-hoc links: {new_next_links}")
        except BaseException as e:
            print(f"\tFailed: {e}")


if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)
