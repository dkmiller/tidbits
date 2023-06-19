"""
https://github.com/dkmiller/notes/blob/main/airbnb/2023/06/12.md
"""

import asyncio
import logging
from argparse import ArgumentParser
from pyppeteer import launch


log = logging.getLogger(__name__)


async def main(port: int):
    browser = await launch()
    page = await browser.newPage()
    url = f"http://localhost:{port}"
    await page.goto(url)
    log.info("Successfully visited %s", url)
    # https://stribny.name/blog/2020/07/creating-website-screenshots-with-python-and-pyppeteer/
    image = await page.screenshot()
    log.info("Screenshotted %d bytes", len(image))
    await browser.close()


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    # https://github.com/dkmiller/tidbits/blob/master/2022/06-29_azdo-conns/add-to-all-subscriptions.py
    asyncio.run(main(args.port))
