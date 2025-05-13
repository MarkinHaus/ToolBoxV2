import os
from collections.abc import Callable
from urllib.parse import urlparse

# Importieren der benötigten Loader und Transformer
from langchain_community.document_loaders import (
    AsyncChromiumLoader,
    AsyncHtmlLoader,
    PlaywrightURLLoader,
    PyPDFLoader,
    ToMarkdownLoader,
    WebBaseLoader,
    WikipediaLoader,
)
from langchain_community.document_transformers import BeautifulSoupTransformer

try:
    from langchain_community.document_loaders import GitLoader

    GitLoader_ac = True
except ImportError as e:
    GitLoader_ac = e
    GitLoader = None


def get_markdown_from_url(url, **kwargs):
    if 'api_key' not in kwargs:
        kwargs['api_key'] = os.environ.get('TOMARKDOWN')
    loader = ToMarkdownLoader(url=url, **kwargs)
    def docs():
        return loader.load()
    return loader, docs


def get_text_from_urls_a_h(urls: list or str, **kwargs):
    if isinstance(urls, str):
        urls = [urls]
    loader = AsyncHtmlLoader(urls, **kwargs)
    def docs():
        return loader.load()
    return loader, docs


def get_text_from_urls_play(urls: list or str, **kwargs):
    if isinstance(urls, str):
        urls = [urls]
    loader = PlaywrightURLLoader(urls, **kwargs)
    def docs():
        return loader.load()
    return loader, docs


def get_wiki_data(query, **kwargs):
    loader = WikipediaLoader(query=query, **kwargs)
    def docs():
        return loader.load()
    return loader, docs


def get_text_from_urls_vue(urls: list or str, tags_to_extract=None):
    if isinstance(urls, str):
        urls = [urls]
    if tags_to_extract is None:
        tags_to_extract = ["p", "li", "div", "a", "h1", "h2", "h3", "h4", "span"]
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=tags_to_extract
    )
    # html2text = Html2TextTransformer()
    # docs_transformed = html2text.transform_documents(docs)
    return loader, docs_transformed


def read_git_repo(repo_path: str,
                  clone_url: str | None = None,
                  branch: str | None = "main",
                  file_filter: Callable[[str], bool] | None = None, ):
    if GitLoader_ac is not True:
        raise ImportError(GitLoader_ac)
    return GitLoader(repo_path, clone_url, branch, file_filter).lazy_load()


def get_pdf_from_url(url: str):
    loader = PyPDFLoader(url)
    def docs():
        return loader.load()
    return loader, docs


def get_data_from_web(urls: list or str, **kwargs):
    loader = WebBaseLoader(urls, **kwargs)
    post_fix = ''
    if isinstance(urls, str):
        post_fix = urls.split('.')[-1]
    if len(post_fix) == 3 and post_fix in ["lxml", "xml"]:
        loader.default_parser = post_fix  # `parser` must be one of html.parser, lxml, xml, lxml-xml, html5lib.
    def docs():
        return loader.load()
    return loader, docs


def _test_valid(docs):
    if len(docs) == 0:
        return False
    if hasattr(docs[0], 'page_content'):
        return "Enable JavaScript" not in docs[0].page_content
    return False


# Definition der Routing-Funktion
def route_url_to_function(url, **kwargs):
    # Bestimmen des URL-Typs
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path.lower()

    if 'to_md' in kwargs:
        return get_markdown_from_url(url, **kwargs)

    # Wikipedia URLs
    if "wikipedia.org" in domain:
        return get_wiki_data(url, **kwargs)

    # PDF Dateien
    elif path.endswith(".pdf"):
        return get_pdf_from_url(url)

    try:
        return get_data_from_web([url], **kwargs)
    except:
        pass

    try:
        loader1, docs1 = get_text_from_urls_vue([url])
        if _test_valid(docs1):
            return loader1, docs1
    except:
        pass

    try:
        loader1, docs1 = get_text_from_urls_a_h([url], **kwargs)
        if _test_valid(docs1()):
            return loader1, docs1
    except:
        pass
    try:
        loader1, docs1 = get_text_from_urls_play([url], **kwargs)
        if _test_valid(docs1()):
            return loader1, docs1
    except:
        pass

        # Hier müsste eine Logik implementiert werden, um die Ergebnisse zu vergleichen und das beste zu wählen
        # Beispiel: return best_loader, best_docs

    # Fallback: Verwende Markdown Loader, wenn alle anderen fehlschlagen
    return get_markdown_from_url(url, **kwargs)
