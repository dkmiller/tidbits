import scrapy


class TestSpider(scrapy.Spider):
    name = "test_spider"
    start_urls = ["https://scrapy.org/"]

    def parse(self, response):
        raise Exception(str(response))
