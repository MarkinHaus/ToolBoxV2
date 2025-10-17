import concurrent
import os
import re
from urllib.parse import quote_plus


import requests
from bs4 import BeautifulSoup


def web_search_serpapi(query: str, max_results: int = 5, api_key: str = None) -> list[dict[str, str]]:
    """
    Web search using SerpAPI (free tier: 100 searches/month)
    Get your free API key at: https://serpapi.com/
    """
    if not api_key:
        print("Please get a free API key from https://serpapi.com/")
        return []

    try:
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": max_results
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        if "organic_results" in data:
            for result in data["organic_results"][:max_results]:
                url_link = result.get("link", "")
                title = result.get("title", "")

                print(f"Processing: {title}")
                markdown_content = url_to_markdown_robust(url_link)

                if markdown_content:
                    results.append({
                        'url': url_link,
                        'title': title,
                        'content': markdown_content
                    })

                #time.sleep(1)  # Be respectful

        return results

    except Exception as e:
        print(f"SerpAPI search error: {e}")
        return []


# Usage:
# results = web_search_serpapi("Python web scraping", api_key="your_serpapi_key")

def web_search_bing(query: str, max_results: int = 5, api_key: str = None) -> list[dict[str, str]]:
    """
    Web search using Bing Search API (free tier: 3,000 queries/month)
    Get your free API key at: https://azure.microsoft.com/en-us/services/cognitive-services/bing-web-search-api/
    """
    if not api_key:
        print("Please get a free API key from Azure Cognitive Services")
        return []

    try:
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {
            "Ocp-Apim-Subscription-Key": api_key
        }
        params = {
            "q": query,
            "count": max_results,
            "textDecorations": False,
            "textFormat": "HTML"
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        if "webPages" in data and "value" in data["webPages"]:
            for result in data["webPages"]["value"][:max_results]:
                url_link = result.get("url", "")
                title = result.get("name", "")

                print(f"Processing: {title}")
                markdown_content = url_to_markdown_robust(url_link)

                if markdown_content:
                    results.append({
                        'url': url_link,
                        'title': title,
                        'content': markdown_content
                    })

                # time.sleep(1)

        return results

    except Exception as e:
        print(f"Bing search error: {e}")
        return []





def is_content_parseable(content: str) -> bool:
    """
    Check if content is properly parsed and readable
    """
    if not content or len(content.strip()) < 50:
        return False

    # Check for too many non-ASCII characters that look like encoding errors
    total_chars = len(content)
    if total_chars == 0:
        return False

    # Count problematic characters
    problematic_chars = 0
    replacement_chars = content.count('�')

    # Check for sequences of garbled characters
    garbled_patterns = [
        r'[ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]{5,}',
        r'[Ã¢Â€Â™Ã¢Â€ÂœÃ¢Â€Â�]{3,}',
        r'[\x80-\xff]{4,}',  # High-byte sequences
        r'[^\x00-\x7F\s]{10,}'  # Too many non-ASCII chars in sequence
    ]

    for pattern in garbled_patterns:
        matches = re.findall(pattern, content)
        problematic_chars += sum(len(match) for match in matches)

    # Calculate ratios
    replacement_ratio = replacement_chars / total_chars
    problematic_ratio = problematic_chars / total_chars

    # Check for readable English content
    english_words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
    english_ratio = len(' '.join(english_words)) / total_chars if english_words else 0

    # Criteria for parseable content
    is_parseable = (
        replacement_ratio < 0.05 and  # Less than 5% replacement chars
        problematic_ratio < 0.15 and  # Less than 15% garbled chars
        english_ratio > 0.3 and  # At least 30% English words
        len(english_words) > 10  # At least 10 English words
    )

    if not is_parseable:
        print("Content failed parseability check:")
        print(f"  Replacement ratio: {replacement_ratio:.1%}")
        print(f"  Problematic ratio: {problematic_ratio:.1%}")
        print(f"  English ratio: {english_ratio:.1%}")
        print(f"  English words: {len(english_words)}")

    return is_parseable


def url_to_markdown_robust(url: str) -> str | None:
    """
    Robust URL to markdown converter with multiple encoding strategies
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Charset': 'utf-8, iso-8859-1;q=0.5',
            'Connection': 'keep-alive'
        }

        response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        response.raise_for_status()

        # Quick content type check
        content_type = response.headers.get('content-type', '').lower()
        if not any(ct in content_type for ct in ['text/html', 'text/plain', 'application/xhtml']):
            print(f"Skipping non-HTML content: {content_type}")
            return None

        # Get raw content
        raw_content = response.content

        # Strategy 1: Try response encoding first if it looks reliable
        decoded_content = None
        used_encoding = None

        response_encoding = response.encoding
        if response_encoding and response_encoding.lower() not in ['iso-8859-1', 'ascii']:
            try:
                decoded_content = response.text
                used_encoding = response_encoding
                # Quick test for encoding quality
                if '�' in decoded_content or not is_mostly_readable(decoded_content[:1000]):
                    decoded_content = None
            except:
                pass

        # Strategy 2: Detect encoding from content
        if not decoded_content:
            try:
                import chardet
                detected = chardet.detect(raw_content)
                if detected and detected.get('confidence', 0) > 0.8:
                    decoded_content = raw_content.decode(detected['encoding'])
                    used_encoding = detected['encoding']
                    if '�' in decoded_content or not is_mostly_readable(decoded_content[:1000]):
                        decoded_content = None
            except ImportError and ModuleNotFoundError:
                print("chardet not installed")
            except:
                pass

        # Strategy 3: Extract encoding from HTML meta tags
        if not decoded_content:
            try:
                # Try UTF-8 first to read meta tags
                temp_content = raw_content.decode('utf-8', errors='ignore')[:2048]
                charset_patterns = [
                    r'<meta[^>]+charset["\'\s]*=["\'\s]*([^"\'>\s]+)',
                    r'<meta[^>]+content[^>]+charset=([^"\'>\s;]+)',
                    r'<\?xml[^>]+encoding["\'\s]*=["\'\s]*([^"\'>\s]+)'
                ]

                for pattern in charset_patterns:
                    match = re.search(pattern, temp_content, re.I)
                    if match:
                        encoding = match.group(1).strip().lower()
                        try:
                            decoded_content = raw_content.decode(encoding)
                            used_encoding = encoding
                            if not ('�' in decoded_content or not is_mostly_readable(decoded_content[:1000])):
                                break
                        except:
                            pass
                        decoded_content = None
            except:
                pass

        # Strategy 4: Try common encodings
        if not decoded_content:
            common_encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
            for encoding in common_encodings:
                try:
                    test_content = raw_content.decode(encoding)
                    if is_mostly_readable(test_content[:1000]) and '�' not in test_content[:1000]:
                        decoded_content = test_content
                        used_encoding = encoding
                        break
                except:
                    continue

        # Strategy 5: Last resort with error handling
        if not decoded_content:
            decoded_content = raw_content.decode('utf-8', errors='replace')
            used_encoding = 'utf-8 (with errors)'

        print(f"Used encoding: {used_encoding}")

        # Parse with BeautifulSoup
        soup = BeautifulSoup(decoded_content, 'html.parser')

        # Remove all unwanted elements aggressively
        unwanted_tags = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe',
                         'form', 'button', 'input', 'noscript', 'meta', 'link', 'svg']
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()

        # Remove elements with unwanted classes/ids
        unwanted_patterns = [
            r'.*ad[s]?[-_].*', r'.*banner.*', r'.*popup.*', r'.*modal.*',
            r'.*cookie.*', r'.*newsletter.*', r'.*social.*', r'.*share.*',
            r'.*comment.*', r'.*sidebar.*', r'.*menu.*', r'.*navigation.*'
        ]

        for pattern in unwanted_patterns:
            for attr in ['class', 'id']:
                for element in soup.find_all(attrs={attr: re.compile(pattern, re.I)}):
                    element.decompose()

        # Find main content with multiple strategies
        main_content = find_main_content(soup)

        if not main_content:
            print("No main content found")
            return None

        # Convert to markdown using multiple strategies
        markdown_content = convert_to_markdown(main_content)

        if not markdown_content:
            print("Markdown conversion failed")
            return None

        # Clean and validate
        cleaned_content = clean_markdown_robust(markdown_content)

        # Final validation
        if not is_content_parseable(cleaned_content):
            print("Content failed parseability check")
            return None

        return cleaned_content

    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None


def is_mostly_readable(text: str) -> bool:
    """Check if text is mostly readable ASCII/common unicode"""
    if not text:
        return False

    readable_chars = sum(1 for c in text if c.isprintable() or c.isspace())
    return readable_chars / len(text) > 0.8


def find_main_content(soup):
    """Find main content using multiple strategies"""

    # Strategy 1: Look for semantic HTML5 elements
    for tag in ['main', 'article']:
        element = soup.find(tag)
        if element and len(element.get_text(strip=True)) > 300:
            return element

    # Strategy 2: Look for common content containers
    content_selectors = [
        '[role="main"]', '.main-content', '#main-content', '.content', '#content',
        '.post-content', '.entry-content', '.article-content', '.blog-content',
        '.story-body', '.article-body', '.post-body'
    ]

    for selector in content_selectors:
        element = soup.select_one(selector)
        if element and len(element.get_text(strip=True)) > 300:
            return element

    # Strategy 3: Find the div with most text content
    divs = soup.find_all('div')
    if divs:
        content_divs = [(div, len(div.get_text(strip=True))) for div in divs]
        content_divs = [(div, length) for div, length in content_divs if length > 300]

        if content_divs:
            content_divs.sort(key=lambda x: x[1], reverse=True)
            return content_divs[0][0]

    # Strategy 4: Use body as fallback
    return soup.find('body')


def convert_to_markdown(element):
    """Convert HTML element to markdown with fallbacks"""

    # Strategy 1: Use html2text
    try:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        h.unicode_snob = True
        h.skip_internal_links = True
        h.inline_links = False
        h.decode_errors = 'ignore'

        markdown = h.handle(str(element))
        if markdown and len(markdown.strip()) > 100:
            return markdown
    except ImportError:
        print("html2text not installed")
    except:
        pass

    # Strategy 2: Extract text with basic formatting
    try:
        text_parts = []

        for elem in element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(elem.name[1])
            text_parts.append('#' * level + ' ' + elem.get_text(strip=True))
            elem.replace_with('[HEADING_PLACEHOLDER]')

        for elem in element.find_all('p'):
            text = elem.get_text(strip=True)
            if text:
                text_parts.append(text)
            elem.replace_with('[PARAGRAPH_PLACEHOLDER]')

        # Get remaining text
        remaining_text = element.get_text(separator='\n', strip=True)

        # Combine all text
        all_text = '\n\n'.join(text_parts)
        if remaining_text:
            all_text += '\n\n' + remaining_text

        return all_text

    except:
        pass

    # Strategy 3: Simple text extraction
    return element.get_text(separator='\n', strip=True)


def clean_markdown_robust(content: str) -> str:
    """Robust markdown cleaning"""
    if not content:
        return ""

    # Remove common encoding artifacts more aggressively
    replacements = {
        '�': '',
        'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€¦': '...',
        'â€"': '-', 'â€"': '--', 'Â': ' ',
        'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
        'â€¢': '•', 'Â·': '·', 'Â«': '«', 'Â»': '»'
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    # Remove lines with too many non-ASCII characters
    lines = content.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            cleaned_lines.append('')
            continue

        # Skip lines that are mostly garbled
        ascii_chars = sum(1 for c in line if ord(c) < 128)
        if len(line) > 10 and ascii_chars / len(line) < 0.7:
            continue

        # Skip navigation/junk lines
        if (len(line) < 3 or
            line.lower() in ['home', 'menu', 'search', 'login', 'register'] or
            re.match(r'^[\W\s]*$', line)):
            continue

        cleaned_lines.append(line)

    # Remove excessive empty lines
    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


def web_search_robust(query: str, max_results: int = 5, max_attempts: int = 15) -> list[dict[str, str]]:
    """
    Robust search that keeps trying until it gets enough good results
    """
    if isinstance(max_results, str):
        if max_results.startswith('"') and max_results.endswith('"') or max_results.startswith("'") and max_results.endswith("'"):
            max_results = max_results[1:-1]
        max_results = int(max_results.strip())
    if isinstance(max_attempts, str):
        if max_attempts.startswith('"') and max_attempts.endswith('"') or max_attempts.startswith("'") and max_attempts.endswith("'"):
            max_attempts = max_attempts[1:-1]
        max_attempts = int(max_attempts.strip())

    def get_more_search_urls(search_query: str, num_urls: int = 15) -> list[dict[str, str]]:
        """Get more URLs than needed so we can filter out bad ones"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'en-US,en;q=0.9',
            }

            # Try DuckDuckGo lite
            search_url = "https://lite.duckduckgo.com/lite/"
            data = {'q': search_query}

            response = requests.post(search_url, data=data, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            results = []

            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                text = link.get_text(strip=True)

                if (href.startswith('http') and
                    'duckduckgo.com' not in href and
                    len(text) > 5 and
                    not any(skip in href.lower() for skip in ['ads', 'shopping', 'images'])):

                    results.append({
                        'url': href,
                        'title': text[:150]
                    })

                    if len(results) >= num_urls:
                        break

            return results

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def get_fallback_urls(search_query: str) -> list[dict[str, str]]:
        """Get fallback URLs from known good sites"""
        encoded_query = quote_plus(search_query)
        fallback_urls = [
            f"https://stackoverflow.com/search?q={encoded_query}",
            f"https://www.reddit.com/search/?q={encoded_query}",
            f"https://medium.com/search?q={encoded_query}",
            f"https://dev.to/search?q={encoded_query}",
            f"https://github.com/search?q={encoded_query}&type=repositories",
            f"https://docs.python.org/3/search.html?q={encoded_query}",
            f"https://realpython.com/?s={encoded_query}",
            f"https://towardsdatascience.com/search?q={encoded_query}",
            f"https://www.geeksforgeeks.org/?s={encoded_query}",
            f"https://hackernoon.com/search?query={encoded_query}"
        ]

        return [
            {'url': url, 'title': f"Search results for '{search_query}'"}
            for url in fallback_urls
        ]

    print(f"Searching for: '{query}' (need {max_results} good results)")

    # Get candidate URLs
    candidate_urls = get_more_search_urls(query, max_attempts)

    if not candidate_urls:
        print("Primary search failed, using fallback URLs...")
        candidate_urls = get_fallback_urls(query)

    print(f"Found {len(candidate_urls)} candidate URLs")

    # Process URLs until we have enough good results
    good_results = []
    processed_count = 0

    def task(candidate):
        markdown_content = url_to_markdown_robust(candidate['url'])
        if markdown_content:
            return {
                'url': candidate['url'],
                'title': candidate['title'],
                'content': markdown_content
            }

    # runn all tasks in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(task, candidate_urls))
        processed_count = len(candidate_urls)

    good_results = [result for result in results if result]

    #for candidate in candidate_urls:
    #    if len(good_results) >= max_results:
    #        break

    #    processed_count += 1
    #    print(f"\n[{processed_count}/{len(candidate_urls)}] Processing: {candidate['title'][:80]}...")

    #    markdown_content = url_to_markdown_robust(candidate['url'])

    #    if markdown_content:
    #        good_results.append({
    #            'url': candidate['url'],
    #            'title': candidate['title'],
    #            'content': markdown_content
    #        })
    #        print(f"✅ Success! Got result {len(good_results)}/{max_results}")
    #    else:
    #        print("❌ Skipped (unparseable or low quality)")

    #    # Small delay to be respectful
    #    time.sleep(1.5)

    print(f"\n🎉 Final results: {len(good_results)} good results out of {processed_count} attempted")
    return good_results


# Main search function
def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """
    Main search function with robust fallbacks
    """
    # Try API searches first if available
    api_keys = {
        'serpapi': os.getenv('SERPAPI_API_KEY'),
        'bing': os.getenv('BING_API_KEY')
    }
    if isinstance(max_results, str):
        if max_results.startswith('"') and max_results.endswith('"') or max_results.startswith("'") and max_results.endswith("'"):
            max_results = max_results[1:-1]
        max_results = int(max_results.strip())
    if api_keys:
        for api_name, api_key in api_keys.items():
            if api_key:
                try:
                    print(f"Trying {api_name.upper()} API...")
                    if api_name == 'serpapi':
                        results = web_search_serpapi(query, max_results, api_key)
                    elif api_name == 'bing':
                        results = web_search_bing(query, max_results, api_key)
                    else:
                        continue

                    if results and len(results) >= max_results:
                        return results
                except Exception as e:
                    print(f"{api_name.upper()} API failed: {e}")

    # Use robust DuckDuckGo search
    return web_search_robust(query, max_results)


# Test function
def robust_search():
    """Test the robust search functionality"""
    query = "Python web scraping best practices"
    results = web_search(query, max_results=3)

    print(f"\n{'=' * 60}")
    print(f"FINAL RESULTS FOR: '{query}'")
    print(f"{'=' * 60}")

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Content length: {len(result['content'])} characters")
        print(f"First 300 chars: {result['content'][:300]}...")

        # Show parseability stats
        content = result['content']
        ascii_ratio = sum(1 for c in content if ord(c) < 128) / len(content)
        print(f"ASCII ratio: {ascii_ratio:.1%}")
        print("-" * 80)


if __name__ == "__main__":
    robust_search()
