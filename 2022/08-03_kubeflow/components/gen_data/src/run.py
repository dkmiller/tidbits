from argparse_dataclass import dataclass
import randomfiletree


@dataclass
class Args:
    output: str
    n_files: int


def main(args: Args):
    print(f"Parsed args: {args}")
    randomfiletree.iterative_gaussian_tree(args.output, nfiles=args.n_files, maxdepth=5)
    print("done")


if __name__ == "__main__":
    args = Args.parse_args()
    main(args)
