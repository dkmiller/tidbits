"""
Utilities related to clustering a collection of URLs with common domain.
"""

import re
from typing import Dict
from urllib.parse import urlparse


def get_groups(urls, suffix: str = r"\d+\.jpg") -> dict:
    parsed = list(map(urlparse, urls))
    groups = map(
        lambda u: re.sub(suffix, "", u.path.split("/")[-1]),
        parsed,
    )
    groups = list(set(groups))
    rv = {}
    for group in groups:
        group_urls = [url for url in urls if group in url]
        rv[group] = group_urls

    # https://stackoverflow.com/a/613218
    rv = dict(sorted(rv.items(), key=lambda item: len(item[1])))
    return rv


def show_groups(groups: Dict[str, list]):
    for index, (group, urls) in enumerate(groups.items()):
        n_urls = len(urls)
        print(f"{index:2} --> {n_urls:2} : {group}")
