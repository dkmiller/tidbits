"""
python run.py  --input-directory ~/tmp/ --input-paths '$.selftext' '$.icon_url' '$.url_overridden_by_dest'

python run.py  --input-directory ~/tmp/ --source-jsonpaths '$.selftext'  --source-key foo --target-key bar --output-directory ~/tmp2/ --output-file-name b.json --target-jsonpath '$.gilded'
"""

from typing import List
from argparse_dataclass import ArgumentParser
from dataclasses import dataclass, field
import json
import jsonpath_ng
import logging
import os
from pathlib import Path


log = logging.getLogger(__name__)


@dataclass
class Args:
    input_directory: str

    output_directory: str
    output_file_name: str

    target_key: str
    target_jsonpath: str

    source_key: str
    source_jsonpaths: List[str] = field(metadata={"nargs": "*"})


def merge_or_select_value(values: list):
    if len(values) == 0:
        rv = ""
    elif len(values) == 1:
        rv = values[0]
    else:
        rv = "".join(values)
    return rv


def select_and_merge_jsonpaths(o, jsonpaths: List[str]):
    path_parsers = list(map(jsonpath_ng.parse, jsonpaths))
    values = []
    for parser in path_parsers:
        o_values = [m.value for m in parser.find(o)]
        o_value = merge_or_select_value(o_values)
        values.append(o_value)
    rv = merge_or_select_value(values)
    return rv


def main(args: Args):
    log.info(f"Running using arguments {args}")
    json_files = Path(args.input_directory).rglob("*.json")

    output_dir = Path(args.output_directory)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / args.output_file_name
    for file_path in json_files:
        with file_path.open("r") as fp:
            lines = fp.readlines()
            for l in lines:
                j = json.loads(l)
                source_value = select_and_merge_jsonpaths(j, args.source_jsonpaths)
                target_value = select_and_merge_jsonpaths(j, [args.target_jsonpath])
                bb = {args.source_key: source_value, args.target_key: target_value}
                jj = json.dumps(bb)
                with output_path.open("a") as out_fp:
                    out_fp.write(jj)
                    out_fp.write(os.linesep)


if __name__ == "__main__":
    parser = ArgumentParser(Args)
    args = parser.parse_args()
    logging.basicConfig(level="INFO")
    main(args)
