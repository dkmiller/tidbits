from argparse_dataclass import dataclass
from directory_tree import display_tree


@dataclass
class Args:
    input: str


def main(args: Args):
    print(f"Parsed args: {args}")
    display_tree(args.input)
    print("done")


if __name__ == "__main__":
    args = Args.parse_args()
    main(args)
