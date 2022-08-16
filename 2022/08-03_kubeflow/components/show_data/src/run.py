from argparse_dataclass import dataclass
from directory_tree import display_tree
import sys
import time


@dataclass
class Args:
    input: str
    sleep: int


def main(args: Args):
    print(f"Raw args: {sys.argv}")
    print(f"Parsed args: {args}")
    display_tree(args.input)

    print(f"Sleeping for {args.sleep} seconds")
    time.sleep(args.sleep)

    print("done")


if __name__ == "__main__":
    args = Args.parse_args()
    main(args)
