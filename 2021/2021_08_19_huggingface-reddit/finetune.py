import argparse
import praw


def emphasize(o):
    print("#" * 100)
    print(o)
    print("#" * 100)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", default="XnkMHEYUujv1wA7EkmToWg")
    parser.add_argument("--client_secret", required=True, type=str)
    parser.add_argument("--post_limit", default=10, type=int)
    parser.add_argument("--subreddit", required=True, type=str)
    parser.add_argument("--user_agent", default="praw.ml")
    return parser


def get_reddit_client(args) -> praw.Reddit:
    rv = praw.Reddit(client_id=args.client_id, client_secret=args.client_secret, user_agent=args.user_agent)
    return rv


def main(args):
    reddit = get_reddit_client(args)
    posts = reddit.subreddit(args.subreddit).top("all", limit=args.post_limit)
    for post in posts:
        emphasize(post.title)
        print(f"\n\n{post.selftext}\n\n")


if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)

