from argparse_dataclass import ArgumentParser
from azureml.core import Run
from azureml.core.run import _SubmittedRun
from dataclasses import dataclass
import logging
import re


log = logging.getLogger(__name__)


@dataclass
class Args:
    token: str
    logging_level: str = "INFO"


def resolve_secret(s: str) -> str:
    if m := re.match(r"secret://(\w+)", s):
        secret_name = m.group(1)
        log.info(f"Resolving secret '{secret_name}' from Azure ML")
        run = Run.get_context()
        if not isinstance(run, _SubmittedRun):
            raise RuntimeError("Secret resolution only works inside Azure ML")
        s = run.get_secret(secret_name)
    return s


def main(args: Args):
    print(resolve_secret(args.token))


if __name__ == "__main__":
    parser = ArgumentParser(Args)
    args = parser.parse_args()
    logging.basicConfig(level=args.logging_level)
    main(args)
