import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from pathlib import Path, PurePosixPath

class GatechSpider(CrawlSpider):
    name = "gatech"
    allowed_domains = ["gatech.edu"]
    start_urls = ["https://www.gatech.edu/"]

    rules = (
        Rule(LinkExtractor(allow_domains="gatech.edu",
                           deny_extensions=[r'\.pdf', r'\.jpg', r'\.png', r'\.zip']),
             callback="save_page",
             follow=True),
    )

    def save_page(self, response):
        # Normalize URL to a filesystem path
        sha = response.url.split("://",1)[1].rstrip("/").replace("/", "_")
        outdir = Path("html")
        outdir.mkdir(exist_ok=True)
        outdir.joinpath(f"{sha}.html").write_bytes(response.body)
        yield {"url": response.url, "sha": sha, "status": response.status}
