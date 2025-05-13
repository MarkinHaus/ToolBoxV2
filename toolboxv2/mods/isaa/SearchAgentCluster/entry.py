# models.py
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ContentCategory(str, Enum):
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    TECHNICAL = "technical"
    REFERENCE = "reference"


class SearchResult(BaseModel):
    content: str
    url: str
    categories: list[ContentCategory]
    relevance_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    depth: int = Field(ge=0, le=3)


class AnalyzedContent(BaseModel):
    key_points: list[str] or str = Field(max_items=3)
    spatial_context: str | None
    temporal_context: str | None
    source_quality: float = Field(ge=0.0, le=1.0)


class SearchSummary(BaseModel):
    main_findings: list[str] = Field(max_items=3)
    cross_references: list[str] = Field(max_items=3)
    confidence_score: float = Field(ge=0.0, le=1.0)


# data_layer.py
import asyncio
from dataclasses import dataclass, field

import aiohttp


@dataclass
class WebCache:
    content: dict[str, str] = field(default_factory=dict)
    visited: set[str] = field(default_factory=set)


class DataLayer:
    def __init__(self, max_depth: int = 3):
        self.cache = WebCache()
        self.max_depth = max_depth
        self.semaphore = asyncio.Semaphore(5)

    async def fetch_content(self, url: str, depth: int) -> str | None:
        if depth > self.max_depth or url in self.cache.visited:
            return None

        async with self.semaphore:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        content = await response.text()
                        self.cache.content[url] = content
                        self.cache.visited.add(url)
                        return content
            except Exception:
                return None


# analysis_layer.py
class ContentAnalyzer:
    def __init__(self, isaa_instance):
        self.isaa = isaa_instance

    def analyze_content(self, content: str, url: str) -> AnalyzedContent:
        analysis_prompt = f"""
        Analyze this content and extract:
        1. Up to 3 key points
        2. Spatial context if any
        3. Temporal context if any
        4. Source reliability (0-1)

        Content: {content[:1000]}
        Source: {url}
        """

        return self.isaa.format_class(
            AnalyzedContent,
            self.isaa.mini_task_completion(analysis_prompt)
        )

    def categorize_result(self, content: str) -> list[ContentCategory]:
        categorization_prompt = f"""
        Determine relevant categories for this content:
        {content[:500]}

        Categories: temporal, spatial, technical, reference
        Return only applicable categories.
        """

        class ContentCategorys(BaseModel):
            contentCategorys: list[ContentCategory]

        categories = self.isaa.format_class(ContentCategorys, categorization_prompt)["contentCategorys"]
        return categories


# search_layer.py
class SemanticSearchEngine:
    def __init__(self, isaa_instance, data_layer: DataLayer, analyzer: ContentAnalyzer):
        self.isaa = isaa_instance
        self.data_layer = data_layer
        self.analyzer = analyzer

    async def search(self, query: str) -> SearchSummary:
        initial_urls = await self._get_initial_urls(query)
        results = []

        for url in initial_urls[:3]:  # Limit to top 3 most relevant URLs
            content = await self.data_layer.fetch_content(url, 0)
            if not content:
                continue

            self.analyzer.analyze_content(content, url)
            categories = self.analyzer.categorize_result(content)

            results.append(SearchResult(
                content=content,
                url=url,
                categories=categories,
                relevance_score=self._calculate_relevance(content, query),
                timestamp=datetime.now(),
                depth=0
            ))

            # Get linked content up to max depth
            linked_urls = self._extract_urls(content)
            for linked_url in linked_urls[:2]:  # Limit to 2 linked URLs per source
                linked_content = await self.data_layer.fetch_content(linked_url, 1)
                if linked_content:
                    self.analyzer.analyze_content(linked_content, linked_url)
                    results.append(SearchResult(
                        content=linked_content,
                        url=linked_url,
                        categories=self.analyzer.categorize_result(linked_content),
                        relevance_score=self._calculate_relevance(linked_content, query),
                        timestamp=datetime.now(),
                        depth=1
                    ))

        return self._create_summary(results)

    async def _get_initial_urls(self, query: str) -> list[str]:
        url_prompt = f"""
        Generate 3 most relevant URLs for searching:
        Query: {query}

        Return only high-quality, authoritative sources.
        """
        class Urls(BaseModel):
            urls : list[str]

        urls = self.isaa.format_class(Urls, url_prompt)["urls"]
        return [url.strip() for url in urls if url.strip()]

    def _calculate_relevance(self, content: str, query: str) -> float:
        relevance_prompt = f"""
        Calculate semantic similarity (0-1):
        Query: {query}
        Content: {content[:500]}
        """

        score = self.isaa.mini_task_completion_format(relevance_prompt, format_=float)
        score = float(score) if score else 0
        return min(max(score, 0.0), 1.0)

    def _create_summary(self, results: list[SearchResult]) -> SearchSummary:
        if not results:
            return SearchSummary(
                main_findings=[],
                cross_references=[],
                confidence_score=0.0
            )

        summary_prompt = f"""
        Create a summary with:
        1. 3 main findings
        2. 3 cross-references between sources
        3. Overall confidence score (0-1)

        Results: {[r.content[:200] for r in results]}
        """

        return self.isaa.format_class(
            SearchSummary,
            self.isaa.mini_task_completion(summary_prompt)
        )

    def _extract_urls(self, content: str) -> list[str]:
        url_prompt = f"""
        Extract up to 2 most relevant linked URLs from:
        {content[:1000]}
        """

        urls = self.isaa.mini_task_completion(url_prompt).split('\n')
        return [url.strip() for url in urls if url.strip()]


# main.py
async def main():
    from toolboxv2 import get_app
    isaa_instance = get_app().get_mod('isaa')
    isaa_instance.init_isaa()
    data_layer = DataLayer(max_depth=3)
    analyzer = ContentAnalyzer(isaa_instance)
    search_engine = SemanticSearchEngine(isaa_instance, data_layer, analyzer)

    query = "What are the effects of climate change on global agriculture?"
    summary = await search_engine.search(query)
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
