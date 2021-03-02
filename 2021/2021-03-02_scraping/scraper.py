import logging
import re
import scrapy
from scrapy.http.response.html import HtmlResponse


log = logging.getLogger(__name__)


counter = 0
distinct_urls = set()


class TestSpider(scrapy.Spider):
    name = "test_spider"
    start_urls = ["https://marriageheat.com/"]

    def parse(self, response: HtmlResponse):
        log.info(f"Parsing {response.url}")

        global counter

        log.info(f"Have visited {counter} URLs, {len(distinct_urls)} distinct.")
        counter += 1
        distinct_urls.add(response.url)

        print(response.text)

        m = re.search(r"Average rating\s+.*", response.text)
        print(m)

        if counter > 2:
            return

        # https://github.com/dkmiller/tidbits/blob/graph-algorithms/2020/2020-12-15_graph-algorithms/Graph.Algorithms/Web.cs
        SELECTOR = "//a[@href]"
        for link in response.xpath(SELECTOR):
            log.debug(f"Found {link}")
            url = link.xpath("@href").extract_first()
            log.debug(f"Got URL {url}")

            if any(u in url for u in TestSpider.start_urls):
                log.debug(f"Queuing {url} to visit next.")
                yield scrapy.Request(url, callback=self.parse)
