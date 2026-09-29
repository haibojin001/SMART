#!/usr/bin/env python3
"""
Lightweight web search (suited to a server environment)

Uses the duckduckgo-search library
- low resource use
- no browser required
- suited to a shared server
- stable
"""

import os
import json
import hashlib
import logging
import time
import random
from pathlib import Path
from typing import Optional, List, Dict

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

CACHE_DIR = Path.home() / ".cache" / "mas-subtitle-search"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ENABLED = True
CACHE_EXPIRY_DAYS = 30

# Search delay
MIN_DELAY = 2.0
MAX_DELAY = 4.0

# ============================================================
# Cache
# ============================================================

def get_cache_path(query: str) -> Path:
    query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
    return CACHE_DIR / f"{query_hash}.json"


def load_from_cache(query: str) -> Optional[str]:
    if not CACHE_ENABLED:
        return None

    cache_path = get_cache_path(query)
    if not cache_path.exists():
        return None

    cache_age_days = (time.time() - cache_path.stat().st_mtime) / 86400
    if cache_age_days > CACHE_EXPIRY_DAYS:
        cache_path.unlink()
        return None

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"✓ Cache hit: {query[:50]}")
            return data.get("result", "")
    except Exception:
        return None


def save_to_cache(query: str, result: str):
    if not CACHE_ENABLED or not result:
        return

    cache_path = get_cache_path(query)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({
                "query": query,
                "result": result,
                "timestamp": time.time()
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Cache write error: {e}")


# ============================================================
# DuckDuckGo search (via the official library)
# ============================================================

def search_duckduckgo(query: str, num_results: int = 3) -> Optional[str]:
    """Search using the duckduckgo-search library."""
    if not DDGS_AVAILABLE:
        logger.error("duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return None

    try:
        # Rate-limiting delay
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        logger.info(f"🔍 DuckDuckGo search: '{query[:60]}' (delay: {delay:.1f}s)")
        time.sleep(delay)

        # Search on the first sentence only: drop the context suffix, 30s timeout
        clean_query = query.split("  ")[0].strip()  # callers separate context with two spaces
        with DDGS(timeout=30) as ddgs:
            results = list(ddgs.text(clean_query, max_results=num_results))

        if not results:
            logger.warning("No results from DuckDuckGo")
            return None

        # Format the output
        output_parts = [f"Search results for: {query}\n"]

        for i, item in enumerate(results, 1):
            title = item.get('title', '')
            snippet = item.get('body', '')
            link = item.get('href', '')

            output_parts.append(f"{i}. {title}")
            if snippet:
                output_parts.append(f"   {snippet}")
            if link:
                output_parts.append(f"   {link}")
            output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return None


# ============================================================
# Common interface
# ============================================================

def web_search_free(query: str, context: str = "", max_results: int = 3) -> str:
    """
    Lightweight web search (suited to a server environment)

    Uses DuckDuckGo HTML search; no browser required.
    """
    full_query = f"{query} {context}".strip()

    # Cache first
    cached = load_from_cache(full_query)
    if cached:
        return cached

    # DuckDuckGo search
    result = search_duckduckgo(full_query, max_results)

    if result:
        save_to_cache(full_query, result)
        return result

    # Failed
    error_msg = f"Search failed for: {query}\n\nPossible reasons:\n"
    error_msg += "- Network connection issue\n"
    error_msg += "- DuckDuckGo rate limiting (try again later)\n"
    error_msg += "- Query blocked or filtered"

    logger.error(error_msg)
    return error_msg


# ============================================================
# Command-line test
# ============================================================

def main():
    """Command-line test."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python web_search_requests.py <query>")
        print("Example: python web_search_requests.py \"What does top of the inning mean in baseball?\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print(f"Query: {query}")
    print("=" * 80)

    result = web_search_free(query, max_results=3)
    print(result)

    print("\n" + "=" * 80)
    print(f"Cache location: {get_cache_path(query)}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
