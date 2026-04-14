"""
src/logic/web_search.py
Tavily search + URL scraping logic.
"""
import os

try:
    from tavily import TavilyClient
    _TAVILY_AVAILABLE = True
except ImportError:
    _TAVILY_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    _SCRAPE_AVAILABLE = True
except ImportError:
    _SCRAPE_AVAILABLE = False


def is_tavily_available() -> bool:
    return _TAVILY_AVAILABLE and bool(os.getenv("TAVILY_API_KEY", ""))


def scrape_url(url: str, max_chars: int = 2000) -> str:
    """Scrape plain text from a URL. Returns empty string on failure."""
    if not _SCRAPE_AVAILABLE:
        return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=6)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())[:max_chars]
    except Exception:
        return ""


def tavily_search(query: str, max_results: int = 4) -> list[dict]:
    """
    Search via Tavily and scrape each result URL for real content.
    Returns list of {title, url, content} dicts.
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key or not _TAVILY_AVAILABLE:
        return []
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=max_results, include_raw_content=False)
        results = response.get("results", [])

        for r in results:
            scraped = scrape_url(r.get("url", ""))
            if len(scraped) > len(r.get("content", "")):
                r["content"] = scraped

        return results
    except Exception as e:
        return []