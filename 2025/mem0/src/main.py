import logging
from argparse import ArgumentParser

from injector import Injector
from strands import Agent

from src.builder import Builder


def main(args):
    injector = Injector(Builder())
    agent = injector.get(Agent)
    # Skip printing the output manually because it already happens automatically.
    _ = agent(prompt=args.prompt)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Who was the president of the United States in 2009?",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    main(args)
