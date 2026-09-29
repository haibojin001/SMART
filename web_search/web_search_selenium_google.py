import time
import logging
import shutil
import tempfile
from typing import List, Dict
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

try:
    import undetected_chromedriver as uc
    UC_AVAILABLE = True
except ImportError:
    UC_AVAILABLE = False
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

import os

_CHROME_PATHS = [
    os.path.expanduser("~/.local/chrome-for-testing/chrome-linux64/chrome"),
    os.path.expanduser("~/.local/chromium/chrome-linux/chrome"),
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
]

_CHROMEDRIVER_PATHS = [
    os.path.expanduser("~/.local/chrome-for-testing/chromedriver-linux64/chromedriver"),
    os.path.expanduser("~/.local/bin/chromedriver"),
]


class GoogleSearchScraper:
    def __init__(self, headless: bool = True, timeout: int = 10):
        self.headless = headless
        self.timeout = timeout
        self.driver = None
        self._tmp_profile = None

    def _init_driver(self):
        if self.driver is not None:
            return

        self._tmp_profile = tempfile.mkdtemp(prefix="chrome_profile_")

        try:
            if UC_AVAILABLE:
                options = uc.ChromeOptions()
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1920,1080")
                options.add_argument("--lang=en-US")
                options.add_argument(f"--user-data-dir={self._tmp_profile}")

                chrome_binary = next((p for p in _CHROME_PATHS if os.path.exists(p)), None)
                if chrome_binary:
                    options.binary_location = chrome_binary

                chromedriver_binary = next((p for p in _CHROMEDRIVER_PATHS if os.path.exists(p)), None)

                self.driver = uc.Chrome(
                    options=options,
                    driver_executable_path=chromedriver_binary,
                    headless=self.headless,
                    use_subprocess=False,
                )
                logger.info("✓ undetected-chromedriver initialized")
            else:
                options = Options()
                if self.headless:
                    options.add_argument("--headless=new")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1920,1080")
                options.add_argument("--lang=en-US")
                options.add_argument(f"--user-data-dir={self._tmp_profile}")
                options.add_argument("--disable-blink-features=AutomationControlled")
                options.add_experimental_option("excludeSwitches", ["enable-automation"])
                options.add_experimental_option("useAutomationExtension", False)

                chrome_binary = next((p for p in _CHROME_PATHS if os.path.exists(p)), None)
                if chrome_binary:
                    options.binary_location = chrome_binary

                chromedriver_binary = next((p for p in _CHROMEDRIVER_PATHS if os.path.exists(p)), None)
                service = Service(chromedriver_binary) if chromedriver_binary else Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
                logger.info("Chrome WebDriver initialized (standard selenium)")

            self.driver.set_page_load_timeout(self.timeout)
        except Exception as e:
            logger.error(f"Chrome WebDriver init failed: {e}")
            raise

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Convert question-style queries to keyword form to reduce CAPTCHA triggers."""
        import re
        q = query.strip().rstrip("?")
        # "What does X mean" / "What is X" → "X meaning/definition"
        m = re.match(r"(?:what does|what is|what are|what's)\s+(.+?)\s+mean(?:s)?(?:\s+in\s+(.+))?$", q, re.I)
        if m:
            term, context = m.group(1), m.group(2)
            return f"{term} meaning{' ' + context if context else ''}"
        # "How does X work" → "X how it works"
        m = re.match(r"how does\s+(.+?)\s+work", q, re.I)
        if m:
            return f"{m.group(1)} how it works"
        # Generic: drop leading question words
        q = re.sub(r"^(what|who|where|when|why|how)\s+(does|is|are|was|were|do|did)\s+", "", q, flags=re.I)
        return q

    def search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        self._init_driver()

        try:
            from urllib.parse import quote_plus
            kw_query = self._normalize_query(query)
            self.driver.get(f"https://www.google.com/search?q={quote_plus(kw_query)}&hl=en")

            try:
                WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.ID, "search"))
                )
            except TimeoutException:
                pass

            # CAPTCHA / rate-limit detection
            if "captcha" in self.driver.page_source.lower() or \
               "unusual traffic" in self.driver.page_source.lower():
                logger.warning("Google CAPTCHA detected, switching to DuckDuckGo for all future searches")
                global _google_captcha_hit
                _google_captcha_hit = True
                return self._search_ddg(query, max_results)

            results = []
            # Anchor on h3 — present in all Google result types (organic, featured, answer box)
            seen_urls = set()
            for h3 in self.driver.find_elements(By.CSS_SELECTOR, "#search h3"):
                if len(results) >= max_results:
                    break
                try:
                    title = h3.text.strip()
                    if not title:
                        continue

                    # URL: h3 is typically a child of the result <a>
                    try:
                        a_elem = h3.find_element(By.XPATH, "ancestor::a[@href][1]")
                        url = a_elem.get_attribute("href")
                    except NoSuchElementException:
                        try:
                            a_elem = h3.find_element(By.XPATH, "following-sibling::a[@href][1]")
                            url = a_elem.get_attribute("href")
                        except NoSuchElementException:
                            continue

                    if not url or not url.startswith("http") or url in seen_urls:
                        continue
                    seen_urls.add(url)

                    # Snippet: walk up 6 levels from h3 to find the result block,
                    # then search downward for snippet text
                    snippet = ""
                    try:
                        # Go up to the result block (tF2Cxc / g / MjjYud)
                        container = h3.find_element(
                            By.XPATH,
                            "ancestor::div[contains(@class,'tF2Cxc') or "
                            "contains(@class,'MjjYud') or contains(@class,'g') or "
                            "contains(@class,'kb0PBd')][1]"
                        )
                        # VwiC3b is the snippet div; also try sibling kb0PBd blocks
                        for sel in ["div.VwiC3b", "div[data-sncf]", "span.aCOpRe",
                                    "div.IsZvec", "div.lEBKkf",
                                    "div.kb0PBd div.VwiC3b"]:
                            try:
                                snippet = container.find_element(By.CSS_SELECTOR, sel).text.strip()
                                if snippet:
                                    break
                            except NoSuchElementException:
                                continue

                        # Last resort: sibling kb0PBd that contains VwiC3b
                        if not snippet:
                            parent = h3.find_element(By.XPATH, "ancestor::div[contains(@class,'kb0PBd')][1]/..")
                            for sib in parent.find_elements(By.CSS_SELECTOR, "div.kb0PBd"):
                                t = sib.text.strip()
                                # snippet blocks don't contain the title text
                                if t and title[:20] not in t and len(t) > 30:
                                    snippet = t
                                    break
                    except NoSuchElementException:
                        pass

                    # Extract snippet from #:~:text= fragment when no snippet div found
                    if not snippet and "#:~:text=" in url:
                        from urllib.parse import unquote
                        frag = url.split("#:~:text=", 1)[1].split("&")[0]
                        parts = unquote(frag).split(",")
                        snippet = " … ".join(p.strip() for p in parts if p.strip())

                    results.append({"title": title, "snippet": snippet, "url": url})
                except Exception:
                    continue

            logger.info(f"Google: {len(results)} results for '{query[:60]}'")
            return results

        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []

    def _search_ddg(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """DuckDuckGo fallback using ddgs library."""
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            for attempt in range(3):
                if attempt > 0:
                    time.sleep(2 * attempt)
                with DDGS(timeout=30) as ddgs:
                    items = list(ddgs.text(query, max_results=max_results))
                if items:
                    break
                logger.warning(f"DDG attempt {attempt+1} returned 0 results, retrying...")

            results = [
                {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
                for r in items
            ]
            logger.info(f"DDG fallback: {len(results)} results for '{query[:60]}'")
            return results
        except Exception as e:
            logger.error(f"DDG fallback failed: {e}")
            return []

    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
        if self._tmp_profile:
            shutil.rmtree(self._tmp_profile, ignore_errors=True)
            self._tmp_profile = None

    def __del__(self):
        self.close()


_scraper_instance = None
_google_captcha_hit = False  # once True, skip Chrome and go straight to DDG


def web_search_free(query: str, max_results: int = 3, **_kwargs) -> str:
    global _scraper_instance, _google_captcha_hit

    if _google_captcha_hit:
        scraper = _scraper_instance or GoogleSearchScraper.__new__(GoogleSearchScraper)
        results = scraper._search_ddg(query, max_results)
        if not results:
            return f"No search results found for: {query}"
        lines = [f"Search Results for: {query}", "=" * 60]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            if r["snippet"]:
                lines.append(f"   {' '.join(r['snippet'].split())}")
            lines.append(f"   URL: {r['url']}\n")
        return "\n".join(lines)

    try:
        if _scraper_instance is None:
            _scraper_instance = GoogleSearchScraper(headless=True, timeout=10)

        results = _scraper_instance.search(query, max_results=max_results)

        if not results:
            return f"No search results found for: {query}"

        lines = [f"Google Search Results for: {query}", "=" * 60]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            if r["snippet"]:
                lines.append(f"   {' '.join(r['snippet'].split())}")
            lines.append(f"   URL: {r['url']}\n")
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"ERROR: Google search failed\n{e}"


def cleanup_scraper():
    global _scraper_instance
    if _scraper_instance:
        _scraper_instance.close()
        _scraper_instance = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for query in [
        "What does 'top of the inning' mean in baseball?",
        "LAPD Detective ranks hierarchy",
    ]:
        print(f"\n{'='*70}\n{query}\n{'='*70}")
        print(web_search_free(query, max_results=2))
        time.sleep(2)

    cleanup_scraper()
