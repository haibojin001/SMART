#!/usr/bin/env python3
"""
Google search via Selenium (a real browser)

Advantages:
- fully emulates a real browser, so Google is less likely to flag it
- renders JavaScript
- stable

Install:
    # Chrome
    pip install selenium webdriver-manager

    # or Firefox (lighter, recommended)
    pip install selenium webdriver-manager
"""

import os
import json
import hashlib
import logging
import time
import random
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

CACHE_DIR = Path.home() / ".cache" / "mas-subtitle-search"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ENABLED = True
CACHE_EXPIRY_DAYS = 30

# Browser choice (auto-detected)
USE_FIREFOX = None  # None=auto-detect, True=Firefox, False=Chrome

# Search delay, to avoid being blocked - deliberately generous
MIN_DELAY = 3.0
MAX_DELAY = 6.0

# WebDriver instance, reused process-wide
_driver = None


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
# Selenium WebDriver
# ============================================================

def detect_browser():
    """Detect which browser is available."""
    import shutil

    # Chrome
    chrome_paths = ['google-chrome', 'chromium', 'chromium-browser', 'chrome']
    for cmd in chrome_paths:
        if shutil.which(cmd):
            return False  # Use Chrome

    # Firefox
    firefox_paths = ['firefox', 'firefox-esr']
    for cmd in firefox_paths:
        if shutil.which(cmd):
            return True  # Use Firefox

    # Fall back to Chrome
    return False


def get_driver():
    """Return the process-wide WebDriver, creating it on first use."""
    global _driver
    global USE_FIREFOX

    if _driver is not None:
        try:
            # Is the driver still alive?
            _driver.current_url
            return _driver
        except Exception:
            _driver = None

    # A privately installed Chrome takes precedence
    custom_chrome = Path.home() / ".local/chrome-for-testing/chrome-linux64/chrome"
    custom_chromedriver = Path.home() / ".local/chrome-for-testing/chromedriver-linux64/chromedriver"

    # With a private Chrome present, use Chrome and skip detection
    if custom_chrome.exists():
        USE_FIREFOX = False
        logger.info(f"Found custom Chrome at: {custom_chrome}")
    # Otherwise detect
    elif USE_FIREFOX is None:
        USE_FIREFOX = detect_browser()
        logger.info(f"Auto-detected browser: {'Firefox' if USE_FIREFOX else 'Chrome'}")

    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        if USE_FIREFOX:
            # Firefox
            try:
                from selenium.webdriver.firefox.options import Options
                from selenium.webdriver.firefox.service import Service
                from webdriver_manager.firefox import GeckoDriverManager

                options = Options()
                options.add_argument('--headless')  # headless
                options.add_argument('--disable-gpu')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')

                # User-Agent
                options.set_preference("general.useragent.override",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0")

                service = Service(GeckoDriverManager().install())
                _driver = webdriver.Firefox(service=service, options=options)
                logger.info("✓ Firefox WebDriver initialized")

            except Exception as e:
                logger.error(f"Firefox WebDriver failed: {e}")
                raise
        else:
            # Chrome (fallback)
            try:
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service

                options = Options()
                # Required in a headless server environment
                options.add_argument('--headless=new')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--disable-software-rasterizer')
                options.add_argument('--disable-extensions')

                # Works around the DevToolsActivePort failure
                options.add_argument('--remote-debugging-port=9222')
                options.add_argument('--disable-setuid-sandbox')

                # Memory and throughput
                options.add_argument('--single-process')
                options.add_argument('--disable-background-networking')
                options.add_argument('--disable-background-timer-throttling')
                options.add_argument('--disable-backgrounding-occluded-windows')
                options.add_argument('--disable-renderer-backgrounding')

                # Reduce automation fingerprinting
                options.add_argument('--disable-blink-features=AutomationControlled')
                options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
                options.add_experimental_option('useAutomationExtension', False)

                # User-Agent
                options.add_argument('user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')

                # Turn off what a search does not need
                prefs = {
                    "profile.default_content_setting_values.notifications": 2,
                    "credentials_enable_service": False,
                    "profile.password_manager_enabled": False,
                    "profile.managed_default_content_settings.images": 2  # skip images: faster
                }
                options.add_experimental_option("prefs", prefs)

                # Use the privately installed Chrome if present
                if custom_chrome.exists():
                    options.binary_location = str(custom_chrome)
                    service = Service(executable_path=str(custom_chromedriver))
                    logger.info(f"Using custom Chrome: {custom_chrome}")
                else:
                    # Otherwise let webdriver-manager fetch a driver
                    from webdriver_manager.chrome import ChromeDriverManager
                    service = Service(ChromeDriverManager().install())
                    logger.info("Using system Chrome via webdriver-manager")

                _driver = webdriver.Chrome(service=service, options=options)

                # Hide the webdriver marker
                _driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                    "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
                })
                _driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

                logger.info("✓ Chrome WebDriver initialized (anti-detection enabled)")

            except Exception as e:
                logger.error(f"Chrome WebDriver failed: {e}")
                raise

        return _driver

    except ImportError as e:
        logger.error(f"Selenium not installed: {e}")
        logger.error("Install with: pip install selenium webdriver-manager")
        raise


def close_driver():
    """Shut the WebDriver down."""
    global _driver
    if _driver is not None:
        try:
            _driver.quit()
        except Exception:
            pass
        _driver = None


# ============================================================
# Google search
# ============================================================

def search_google_selenium(query: str, num_results: int = 3) -> Optional[str]:
    """
    Search Google through Selenium.

    Emulates a real browser, so it is less likely to be flagged.
    """
    try:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver = get_driver()

        # Randomised delay
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        logger.info(f"🔍 Google search (Selenium): '{query[:60]}' (delay: {delay:.1f}s)")
        time.sleep(delay)

        # Build the search URL
        from urllib.parse import quote_plus
        encoded_query = quote_plus(query)
        search_url = f"https://www.google.com/search?q={encoded_query}&num={num_results + 5}&hl=en"

        # Visit the home page first, as a person would
        try:
            driver.get("https://www.google.com")
            time.sleep(random.uniform(0.5, 1.5))
        except Exception:
            pass

        # Then the results page
        driver.get(search_url)

        # Wait for the results container
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "search"))
            )
        except Exception:
            logger.warning("Search results container not found, continuing anyway...")

        # Pause as a reader would
        time.sleep(random.uniform(1.5, 3.0))

        # CAPTCHA check
        page_source = driver.page_source.lower()
        if "captcha" in page_source or "unusual traffic" in page_source:
            logger.error("Google CAPTCHA detected!")
            return None

        # Extract the results
        results = []

        # Strategy 1: <div class="g"> or data-sokoban-container
        try:
            # Results normally sit under id="search"
            search_divs = driver.find_elements(By.CSS_SELECTOR, '#search div.g, div[data-sokoban-container]')

            for div in search_divs[:num_results * 2]:  # over-fetch: some will be filtered out
                try:
                    # title
                    h3_elem = div.find_element(By.TAG_NAME, 'h3')
                    title = h3_elem.text.strip()

                    # link
                    link = ""
                    try:
                        a_elem = div.find_element(By.TAG_NAME, 'a')
                        link = a_elem.get_attribute('href')
                    except Exception:
                        pass

                    # snippet
                    snippet = ""
                    try:
                        # the class name varies by result type
                        for selector in ['div.VwiC3b', 'div.IsZvec', 'span.aCOpRe', 'div.yXK7lf']:
                            try:
                                snippet_elem = div.find_element(By.CSS_SELECTOR, selector)
                                snippet = snippet_elem.text.strip()
                                if snippet:
                                    break
                            except Exception:
                                continue
                    except Exception:
                        pass

                    if title and len(title) > 3:  # require actual content
                        results.append({
                            'title': title,
                            'snippet': snippet,
                            'link': link
                        })

                except Exception as e:
                    continue

        except Exception as e:
            logger.warning(f"Method 1 failed: {e}")

        # Strategy 2: every h3 on the page
        if len(results) < num_results:
            try:
                h3_elements = driver.find_elements(By.TAG_NAME, 'h3')

                for h3 in h3_elements:
                    try:
                        title = h3.text.strip()
                        if not title or len(title) < 3:
                            continue

                        # the link lives on the parent
                        link = ""
                        try:
                            parent = h3.find_element(By.XPATH, './..')
                            a_elem = parent.find_element(By.TAG_NAME, 'a')
                            link = a_elem.get_attribute('href')
                        except Exception:
                            pass

                        # the snippet is in the parent or a sibling
                        snippet = ""
                        try:
                            parent = h3.find_element(By.XPATH, './../..')
                            divs = parent.find_elements(By.TAG_NAME, 'div')
                            for div in divs:
                                text = div.text.strip()
                                if len(text) > 20 and text != title:
                                    snippet = text
                                    break
                        except Exception:
                            pass

                        if title:
                            results.append({
                                'title': title,
                                'snippet': snippet,
                                'link': link
                            })

                    except Exception:
                        continue

            except Exception as e:
                logger.warning(f"Method 2 failed: {e}")

        # De-duplicate on title
        seen_titles = set()
        unique_results = []
        for r in results:
            if r['title'] not in seen_titles:
                seen_titles.add(r['title'])
                unique_results.append(r)

        if not unique_results:
            logger.warning("No results extracted from page")
            return None

        # Format the output
        output_parts = [f"Search results for: {query}\n"]

        for i, item in enumerate(unique_results[:num_results], 1):
            output_parts.append(f"{i}. {item['title']}")
            if item['snippet']:
                output_parts.append(f"   {item['snippet']}")
            if item['link']:
                output_parts.append(f"   {item['link']}")
            output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"Selenium search failed: {e}")
        return None


# ============================================================
# Common interface
# ============================================================

def web_search_free(query: str, context: str = "", max_results: int = 3) -> str:
    """
    Free web search via Selenium and Google.

    Cached, with the failure paths handled.
    """
    full_query = f"{query} {context}".strip()

    # Cache first
    cached = load_from_cache(full_query)
    if cached:
        return cached

    # Selenium search
    result = search_google_selenium(full_query, max_results)

    if result:
        save_to_cache(full_query, result)
        return result

    # Failed
    error_msg = f"Search failed for: {query}\n\nPossible reasons:\n"
    error_msg += "- Selenium WebDriver not installed (pip install selenium webdriver-manager)\n"
    error_msg += "- Browser driver download failed\n"
    error_msg += "- Google CAPTCHA (try again later)\n"
    error_msg += "- Network connection issue"

    logger.error(error_msg)
    return error_msg


# ============================================================
# Command-line test
# ============================================================

def main():
    """Command-line test."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python web_search_selenium.py <query>")
        print("Example: python web_search_selenium.py \"What does top of the inning mean in baseball?\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print(f"Query: {query}")
    print("=" * 80)

    try:
        result = web_search_free(query, max_results=3)
        print(result)

        print("\n" + "=" * 80)
        print(f"Cache location: {get_cache_path(query)}")
        print("Note: the WebDriver stays up, which makes later searches faster")

    finally:
        # Closing the driver on exit
        # Left open on purpose: reusing it makes the next search much faster
        pass  # close_driver()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted; closing the browser...")
        close_driver()
    except Exception as e:
        print(f"\nError: {e}")
        close_driver()
