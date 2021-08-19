import argparse
from gfycat.client import GfycatClient
from lxml import html
import praw
import requests
from urllib.parse import urlparse


def emphasize(o):
    print("#" * 80)
    print(o)
    print("#" * 80)


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
    rv = praw.Reddit(client_id=args.reddit_id, client_secret=args.reddit_secret, user_agent=args.user_agent)
    return rv


def main(args):
    reddit = get_reddit_client(args)
    gfycat = get_gfycat_client(args)
    posts = reddit.subreddit(args.subreddit).top("all", limit=args.limit)

    for post in posts:
        emphasize(post.title)
        print(f"\tFound {post.url}")
        if args.output:
            with open(args.output, "a") as f:
                f.write(f"{post.title}\n    {post.url}\n")
        parsed_url = urlparse(post.url)

        if parsed_url.netloc == "gfycat.com":
            name = parsed_url.path.split("/")[-1]
            try:
                print(gfycat.query_gfy(name))
            except BaseException as e:
                print(f"\tFailed: {e}")
        
        try:
            r = requests.get(post.url)
            # from lxml import html
            # tree = html.parse(r.raw)
            tree = html.fromstring(r.text)
            # print(tree)
            next_links = [n for n in [c.attrib.get("content", None) for c in tree.getchildren()[0].getchildren()] if n and n.endswith(".mp4")]
            if next_links:
                print(next_links)
        except BaseException as e:
            print(f"Failed: {e}")
        
        # if "gfycat" in post.url:


        # r = requests.get(post.url, stream=True)
        # https://stackoverflow.com/a/33511557
        # TODO: consider lxml.
        # t = lxml.html.parse(r.raw)
        # print(t)
        # https://www.reddit.com/r/learnpython/comments/abt1kp/downloading_gifs_on_reddit_from_gfycat/
        # print(t.xpath("//source"))

        # r = requests.get(post.url)
        # print(r.text)
        # soup = BeautifulSoup(r.text, "html.parser")


if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)

