import logging
import os
import scrapy
from scrapy.http.response.html import HtmlResponse


log = logging.getLogger(__name__)


class TestSpider(scrapy.Spider):
    name = "test_spider"
    start_urls = [os.environ["START_URL"]]

    custom_settings = {
        # Don't hit pages too often: https://stackoverflow.com/a/8772954 .
        "DOWNLOAD_DELAY": 2
    }

    def parse(self, response: HtmlResponse):
        log.info(f"Parsing {response.url}")

        title = response.xpath("//title/text()").extract_first()
        log.info(f"Visiting {title}")

        start_url = TestSpider.start_urls[0]
        p_url = f"{start_url}?p="

        for link_href in response.xpath("//link[@href]"):
            url = link_href.xpath("@href").extract_first()

            if p_url in url:
                log.info(f"Recording page ID URL: {url}")
                yield {"title": title, "long_url": response.url, "short_url": url}

        # https://github.com/dkmiller/tidbits/blob/graph-algorithms/2020/2020-12-15_graph-algorithms/Graph.Algorithms/Web.cs
        for link in response.xpath("//a[@href]"):
            url = link.xpath("@href").extract_first()

            if start_url in url and ("#comment-" not in url) and ("mailto:" not in url):
                log.info(f"Queuing {url} to visit.")
                yield scrapy.Request(url, callback=self.parse)
