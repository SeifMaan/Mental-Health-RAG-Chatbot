# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.
import scrapy


class MentalHealthFullSpider(scrapy.Spider):
    name = "scraper"
    allowed_domains = ["mentalhealth.org.uk"]
    start_urls = [
        "https://www.mentalhealth.org.uk/explore-mental-health/a-z-topics",
        "https://www.mentalhealth.org.uk/publications",
        "https://www.mentalhealth.org.uk/your-mental-health",
        "https://www.mentalhealth.org.uk/mental-health-tips",
        "https://www.mentalhealth.org.uk/research-impact",
    ]

    visited_urls = set()  # Prevent duplicate crawls

    def parse(self, response):
        topic_links = response.css("div.c-glossary-item a::attr(href)").getall()
        print(f"Found {len(topic_links)} topic links: {topic_links[:5]}...")  # Debug, show first 5 links

        for link in topic_links:
            full_url = response.urljoin(link)
            yield response.follow(full_url, self.parse_topic)

    def parse_topic(self, response):
        title = response.css("div.block-mhf-theme-page-title h1 span::text").get()
        
        if not title:
            title = response.css("h1 span::text").get()  # Fallback to h1 span if specific div fails
        if not title:
            title = response.url.split("/")[-1].replace("-", " ").title()  # Fallback to URL-derived title
        # Updated paragraph selector with fallback
        paragraphs = response.css("div.c-p-summary p::text").getall()
        if not paragraphs:
            paragraphs = response.css("p::text").getall()  # Broader fallback

        print(f"URL: {response.url}, Title: {title}, Paragraphs count: {len(paragraphs)}")  # Debug

        if title:  # Relaxed condition to yield even if paragraphs are empty
            yield {
                "source": "mentalhealth.org.uk",
                "url": response.url,
                "title": title.strip(),
                "content": " ".join(p.strip() for p in paragraphs if p.strip()) if paragraphs else "No content"
}
