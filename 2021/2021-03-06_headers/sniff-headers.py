import argparse
from haralyzer import HarParser, HarPage
from http.cookies import SimpleCookie
import json
import logging
from typing import Dict
from urllib.parse import parse_qs, urlparse


log = logging.getLogger(__name__)


interesting_values = set()


def log_interesting_value(kind: str, val: str):
    if val in interesting_values:
        log.debug(f"Ignoring duplicate {val}")
    else:
        interesting_values.add(val)
        log.info(f"{kind}: {val}")


def is_interesting_header(header) -> bool:
    name = header["name"].lower()
    is_known_header = name in ["authorization", "cookie"]

    matches = "token" in name or "session" in name
    return is_known_header or matches


def sniff_authorization(auth: str):
    split = auth.split(" ")
    if split[0] == "Bearer":
        log_interesting_value("Bearer token", split[1])
    else:
        log.debug(f"Ignoring {auth}")


def sniff_cookie(cookie: str):
    log.info(f"Sniffing cookie {cookie[:20]}...")
    cookie = SimpleCookie(cookie)
    for k, v in cookie.items():
        log_interesting_value(f"Cookie {k}", v.value)


def sniff_header(header: Dict[str, str]):
    name = header["name"].lower()
    if name == "authorization":
        sniff_authorization(header["value"])
    elif name == "cookie":
        sniff_cookie(header["value"])
    elif is_interesting_header(header):
        print(f"\t\t{header}")
    else:
        log.debug(f"ignore {header['name']}")


def sniff_url(url: str):
    parsed_url = urlparse(url)
    log.info(f"Sniffing {parsed_url.hostname}")
    query = parse_qs(parsed_url.query)
    for key, value in query.items():
        print(f"\tQuery: {key} -> {value}")


def sniff_page(page: HarPage):
    sniff_url(page.url)
    for entry in page.entries:
        log.info(f"Sniffing {entry.url}")
        for header in entry.request.headers:
            sniff_header(header)


def main(args):
    logging.basicConfig(level=args.level)
    with open(args.archive, "r", encoding="utf-8") as f:
        body = json.load(f)
    har_parser = HarParser(body)

    from visitors import HttpArchiveVisitor

    visitor = HttpArchiveVisitor()
    visitor.visit(har_parser)

    visitor.summarize()

    # for page in har_parser.pages:
    #     sniff_page(page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--level", default="INFO")
    args = parser.parse_args()
    main(args)
