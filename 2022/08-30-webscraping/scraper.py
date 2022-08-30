import aiohttp
import asyncio
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
import logging
import re
from threading import Lock

# https://stackoverflow.com/a/56434301
import nest_asyncio

nest_asyncio.apply()


log = logging.getLogger(__name__)


@dataclass
class WebScraper:
    base: str
    regex: str
    session: aiohttp.ClientSession

    max_concurrency: int = 10
    sleep: int = 3

    lock: Lock = Lock()
    active_visits: int = 0
    collected: set = field(default_factory=set)
    queue: list = field(default_factory=list)
    visited: set = field(default_factory=set)

    async def scrape(self):
        while True:
            with self.lock:
                active_visits = self.active_visits
                n_dequeue = min(len(self.queue), self.max_concurrency - active_visits)
                # https://stackoverflow.com/a/33626754
                queue = self.queue[:n_dequeue]
                self.queue = self.queue[n_dequeue:]

            if queue:
                log.info(f"Found {len(queue)} URLs in the queue; visiting them")

                with self.lock:
                    self.active_visits += len(queue)

                future = asyncio.gather(*[self.visit(u) for u in queue])
                # https://stackoverflow.com/a/44630895
                loop = asyncio.get_running_loop()
                loop.run_until_complete(future)
            elif active_visits:
                log.info(
                    f"{active_visits} active visits and nothing in the queue; sleeping {self.sleep} seconds"
                )
                await asyncio.sleep(self.sleep)
            else:
                log.info("No active visits and nothing in the queue; done!")
                return

    async def visit(self, url: str):
        with self.lock:
            if url in self.visited:
                self.active_visits -= 1
                return
            self.visited.add(url)

        try:
            async with self.session.get(url) as r:
                text = await r.text()
                soup = BeautifulSoup(text, "html.parser")
                links = [x.get("href") for x in soup.find_all("a")]
                collected_links = [
                    re.sub(self.regex, "", l) for l in links if re.search(self.regex, l)
                ]

                links = [x for x in links if x and self.base in x]
        except Exception as e:
            log.warning(f"Encountered error {e} with {url}")
            links = []
            collected_links = []

        with self.lock:
            self.active_visits -= 1
            self.collected.update(collected_links)
            links = [l for l in links if l not in self.visited]
            if links:
                log.info(f"Found {len(links)} new links in {url}")
            self.queue.extend(links)


async def main(base: str, regex: str):
    async with aiohttp.ClientSession() as session:
        scraper = WebScraper(base, regex, session)
        scraper.queue.append(base)
        await scraper.scrape()


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    asyncio.run(main("https://greyenlightenment.com/", "wordpress"))
