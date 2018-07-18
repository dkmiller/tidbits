'''
Baby (test) Reddit bot.
'''

import argparse
import praw

def main(args):
    print('Howdy!')
    reddit = praw.Reddit('baby',client_id=args.id, client_secret=args.secret)
    subreddit = reddit.subreddit('learnpython')

    for submission in subreddit.hot(limit=5):
        print(submission)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the baby Reddit bot.')
    parser.add_argument('--id', help='Client ID', required=True)
    parser.add_argument('--secret', help='Client secret', required=True)
    args = parser.parse_args()
    main(args)

