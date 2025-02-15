import json
import time

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote


class GoogleSearch:
    def __init__(self, query):
        self.query = query
        self.url = f"https://www.google.com/search?q={query}"

    def search(self):
        response = requests.get(self.url)
        if response.status_code != 200:
            time.sleep(2)
            response = requests.get(self.url)
        soup = BeautifulSoup(response.content, 'html.parser')
        results = []
        for result in soup.find_all('div', class_='r'):
            title = result.find('h3').text
            link = result.find('a')['href']
            results.append({'title': title, 'link': link})
        return results


class BingSearch:
    def __init__(self, query):
        self.query = query
        self.url = f"https://www.bing.com/search?q={query}"

    def search(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.content, 'html.parser')
        results = []
        for result in soup.find_all('li', class_='b_algo'):
            title = result.find('h2').text
            link = result.find('a')['href']
            results.append({'title': title, 'link': link})
        return results


class SearchEngineInterface:
    def __init__(self, query):
        self.query = query.replace(' ', '+')
        self.search_engines = [GoogleSearch(quote(query)), BingSearch(quote(query))]

    def search(self):
        results = []
        for search_engine in self.search_engines:
            for res in search_engine.search():
                if res in results:
                    continue
                results.append(res)

        return results

import asyncio
from typing import List
from crawl4ai import BrowserConfig, CrawlerRunConfig, WebCrawler, AsyncWebCrawler
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import os
import sys
import psutil
import asyncio


async def crawl_sequential(urls: List[str]):

    browser_config = BrowserConfig(
        headless=True,
        # For better performance in Docker or low-memory environments:
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],

    )

    crawl_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator()
    )

    # Create the crawler (opens the browser)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    results = []
    try:
        session_id = "session1"  # Reuse the same session across all URLs
        for url in urls:
            result = await crawler.arun(
                url=url,
                config=crawl_config,
                session_id=session_id
            )
            if result.success:
                # E.g. check markdown length
                results.append(result.markdown_v2.raw_markdown)
            else:
                print(f"Failed: {url} - Error: {result.error_message}")
    finally:
        # After all URLs are done, close the crawler (and the browser)
        await crawler.close()

    return results


async def web_search(query: str, mas_text_summaries=None):
    search_interface = SearchEngineInterface(query)
    results = search_interface.search()
    urls = []
    for result in results:
        print(f"Title: {result['title']}, Link: {result['link']}")
        urls.append(result['link'])

    c_up = lambda x: '\n'.join([_ for _ in x.split('\n') if not _.strip().startswith('*')])

    if len(urls) == 0:

        query_ = quote(query.replace(' ', '+'))

        start_urls = [f"https://www.bing.com/search?q={query_}", f"https://www.google.com/search?q={query_}"]

        start_data = await crawl_sequential(start_urls)

        result = c_up('\n'.join(start_data))
        if mas_text_summaries:
               result = mas_text_summaries(result, ref=query)
        return result

    data = await crawl_sequential(urls)
    final_result = c_up('\n'.join(data))
    if mas_text_summaries:
        final_result = mas_text_summaries(final_result, ref=query)
    return final_result


if __name__ == "__main__":
    asyncio.run(web_search("Wer ist Lanna Idris"))
