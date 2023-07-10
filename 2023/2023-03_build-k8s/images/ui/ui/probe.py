"""
https://github.com/dkmiller/notes/blob/main/airbnb/2023/06/12.md
"""

import asyncio
import logging
import os
from argparse import ArgumentParser
from typing import List

from pyppeteer import launch

log = logging.getLogger(__name__)


async def probe(url: str):
    log.info("Starting probe")
    # https://github.com/puppeteer/puppeteer/issues/3698#issuecomment-504648941
    # https://github.com/isholgueras/chrome-headless/issues/1#issuecomment-924713127
    browser = await launch(
        {
            "dumpio": True,
            "args": ["--no-sandbox", "--disable-gpu", "--disable-software-rasterizer"],
            "headless": True,
        }
    )
    log.info("Launched browser %s", browser)
    page = await browser.newPage()
    await page.goto(url)
    log.info("Successfully visited %s", url)
    # https://stribny.name/blog/2020/07/creating-website-screenshots-with-python-and-pyppeteer/
    image = await page.screenshot()
    log.info("Screenshotted %d bytes", len(image))
    await browser.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("--logfile", type=str, default="/proc/1/fd/1")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8501",
        help="Alternate: http://host.docker.internal:8501",
    )
    args = parser.parse_args()

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if os.path.isfile(args.logfile):
        # https://stackoverflow.com/a/46098711
        handlers.append(logging.FileHandler(args.logfile))

    # https://stackoverflow.com/a/46098711
    logging.basicConfig(level="INFO", handlers=handlers)

    # https://github.com/dkmiller/tidbits/blob/master/2022/06-29_azdo-conns/add-to-all-subscriptions.py
    asyncio.run(probe(args.url))
