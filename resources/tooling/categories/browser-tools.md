# Browser Automation Tools for Agents
> Browser tools give agents eyes and hands on the web -- but anti-bot detection, auth flows, and flaky selectors make them the hardest tools to keep reliable in production.

## What It Is

Browser automation tools allow agents to interact with web pages: navigating URLs, clicking elements, filling forms, extracting text, and taking screenshots. They bridge the gap between structured APIs (which many services don't have) and the web interfaces that humans use. Common implementations use Playwright or Puppeteer under the hood, controlled by the agent through tool calls.

There are two paradigms:
1. **DOM-based interaction**: The agent receives HTML/DOM structure and specifies elements by selectors
2. **Screenshot-based interaction (vision)**: The agent sees a screenshot and specifies click coordinates

## How It Works

### Architecture

```
LLM decides to browse a page
  |
  v
[Browser Tool] -- e.g., navigate_to_url, click_element, extract_text
  |
  v
[Browser Controller] -- Playwright/Puppeteer orchestration
  |
  v
[Headless Browser] -- Chromium instance
  |
  |-- Stealth mode (anti-bot evasion)
  |-- Proxy rotation (IP diversity)
  |-- Cookie/session management
  |-- Screenshot capture
  |-- Network interception
  |
  v
[Page State]
  |-- DOM tree (simplified for LLM context)
  |-- Screenshot (for vision-based agents)
  |-- Extracted text/data
  |
  v
Return to LLM
```

### DOM-Based vs Screenshot-Based

| Aspect | DOM-Based | Screenshot-Based |
|--------|-----------|-----------------|
| LLM input | Simplified HTML/accessibility tree | Screenshot image |
| Precision | High (exact selectors) | Medium (coordinate-based) |
| Token cost | Medium (~500-2000 tokens per page) | High (image tokens) |
| Dynamic content | Handles JS-rendered content | Sees final rendered page |
| Anti-bot detection | Lower risk (no visual rendering) | Higher risk (renders like human) |
| Complex layouts | Struggles with visual-only cues | Better at visual understanding |
| Best for | Structured pages, forms | Visual-heavy sites, CAPTCHAs |

## Production Implementation

### Playwright-Based Browser Tool

```python
"""Production browser automation tool using Playwright."""
from playwright.async_api import async_playwright, Browser, Page, TimeoutError
from typing import Optional
import asyncio
import base64
import logging

logger = logging.getLogger(__name__)

# Configuration
NAVIGATION_TIMEOUT = 15000  # 15s
ACTION_TIMEOUT = 5000       # 5s
MAX_CONTENT_LENGTH = 5000   # Characters returned to LLM


class BrowserSession:
    """Manages a persistent browser session for an agent."""

    def __init__(self):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

    async def start(self):
        """Initialize browser with stealth settings."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        context = await self._browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            # Block unnecessary resources for speed
            bypass_csp=True,
        )
        self._page = await context.new_page()

        # Block heavy resources
        await self._page.route(
            "**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf}",
            lambda route: route.abort(),
        )

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    @property
    def page(self) -> Page:
        return self._page


class BrowserTools:
    """Browser automation tools for LLM agents."""

    def __init__(self):
        self.session = BrowserSession()
        self._initialized = False

    async def _ensure_initialized(self):
        if not self._initialized:
            await self.session.start()
            self._initialized = True

    # --- Tool: Navigate ---
    async def navigate_to_url(self, url: str) -> dict:
        """Navigate to a URL and return the page content.

        Use this to open a web page. Returns the page title and 
        a simplified text version of the page content.
        """
        await self._ensure_initialized()

        try:
            response = await self.session.page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=NAVIGATION_TIMEOUT,
            )

            if response and response.status >= 400:
                return {
                    "success": False,
                    "error_type": "navigation_error",
                    "message": f"Page returned HTTP {response.status}.",
                }

            title = await self.session.page.title()
            # Extract readable text content
            content = await self.session.page.evaluate("""
                () => {
                    // Remove script and style elements
                    const scripts = document.querySelectorAll('script, style, noscript');
                    scripts.forEach(s => s.remove());
                    return document.body.innerText.substring(0, 5000);
                }
            """)

            return {
                "success": True,
                "data": {
                    "url": self.session.page.url,
                    "title": title,
                    "content": content[:MAX_CONTENT_LENGTH],
                },
            }

        except TimeoutError:
            return {
                "success": False,
                "error_type": "timeout",
                "message": f"Navigation to {url} timed out after {NAVIGATION_TIMEOUT/1000}s.",
            }

    # --- Tool: Click Element ---
    async def click_element(self, selector: str) -> dict:
        """Click an element on the current page.

        Use this to click buttons, links, or interactive elements.
        Provide a CSS selector (e.g., 'button.submit', '#login-btn', 
        'a[href="/pricing"]') or text selector ('text=Sign Up').
        """
        await self._ensure_initialized()

        try:
            await self.session.page.click(selector, timeout=ACTION_TIMEOUT)
            # Wait for navigation or dynamic content
            await self.session.page.wait_for_load_state("domcontentloaded", timeout=5000)

            return {
                "success": True,
                "data": {
                    "clicked": selector,
                    "current_url": self.session.page.url,
                    "page_title": await self.session.page.title(),
                },
            }

        except TimeoutError:
            return {
                "success": False,
                "error_type": "not_found",
                "message": f"Element '{selector}' not found on page or not clickable. "
                           f"Try a different selector.",
                "suggestions": [
                    "Use take_screenshot to see the current page.",
                    "Try a text-based selector: text='Button Text'.",
                    "Try a more specific CSS selector.",
                ],
            }

    # --- Tool: Fill Form Field ---
    async def fill_field(self, selector: str, value: str) -> dict:
        """Type text into a form field.

        Use this to fill input fields, textareas, or search boxes.
        Clears any existing text before typing.
        """
        await self._ensure_initialized()

        try:
            await self.session.page.fill(selector, value, timeout=ACTION_TIMEOUT)
            return {
                "success": True,
                "data": {"field": selector, "value_entered": value},
            }
        except TimeoutError:
            return {
                "success": False,
                "error_type": "not_found",
                "message": f"Input field '{selector}' not found.",
            }

    # --- Tool: Extract Text ---
    async def extract_text(self, selector: str = "body") -> dict:
        """Extract text content from the current page or a specific element.

        Use this to read page content. Provide a CSS selector to 
        extract from a specific section, or omit for full page text.
        """
        await self._ensure_initialized()

        try:
            text = await self.session.page.text_content(selector, timeout=ACTION_TIMEOUT)
            return {
                "success": True,
                "data": {
                    "selector": selector,
                    "text": (text or "")[:MAX_CONTENT_LENGTH],
                },
            }
        except TimeoutError:
            return {
                "success": False,
                "error_type": "not_found",
                "message": f"Element '{selector}' not found on page.",
            }

    # --- Tool: Take Screenshot ---
    async def take_screenshot(self) -> dict:
        """Take a screenshot of the current page.

        Use this to see what the page looks like, especially when
        you're unsure about the layout or available elements.
        Returns a base64-encoded PNG image.
        """
        await self._ensure_initialized()

        screenshot = await self.session.page.screenshot(
            type="png",
            full_page=False,  # Viewport only for manageable size
        )

        return {
            "success": True,
            "data": {
                "image_base64": base64.b64encode(screenshot).decode(),
                "format": "png",
                "url": self.session.page.url,
            },
        }

    # --- Tool: Get Page Links ---
    async def get_page_links(self) -> dict:
        """Get all links on the current page.

        Returns a list of links with their text and URLs.
        Useful for discovering navigation options.
        """
        await self._ensure_initialized()

        links = await self.session.page.evaluate("""
            () => {
                const links = Array.from(document.querySelectorAll('a[href]'));
                return links.slice(0, 50).map(a => ({
                    text: a.innerText.trim().substring(0, 100),
                    href: a.href,
                })).filter(l => l.text && l.href);
            }
        """)

        return {
            "success": True,
            "data": {"links": links, "count": len(links)},
        }
```

## Decision Tree: When to Use Browser Tools

```
Does the target service have a structured API?
  |
  YES --> Use API integration tool (more reliable, faster)
  |
  NO --> Is the data publicly available on a web page?
          |
          YES --> Does the page require JavaScript rendering?
          |         |
          |         YES --> Browser tool (Playwright)
          |         NO  --> HTTP GET + HTML parser (simpler, faster)
          |
          NO --> Does it require authentication?
                  |
                  YES --> Browser tool with auth flow
                  |       (But consider: can you get API access instead?)
                  NO  --> Browser tool for dynamic content
```

## When NOT to Use Browser Tools

- **An API exists**: Always prefer structured APIs over browser automation
- **Static content**: Use `httpx` + BeautifulSoup for simple HTML scraping
- **High-frequency operations**: Browser automation is slow (2-15s per action)
- **Sites with aggressive anti-bot**: CAPTCHAs, Cloudflare challenges -- these break often
- **Login-gated content**: Maintaining browser sessions across agent runs is fragile

## Tradeoffs

| Aspect | Browser Tools | Direct API | HTTP Scraping |
|--------|--------------|------------|---------------|
| Reliability | Low (fragile selectors) | High | Medium |
| Speed | Slow (2-15s/action) | Fast (<500ms) | Fast (<1s) |
| Maintenance | High (UI changes break it) | Low | Medium |
| JS rendering | Full support | N/A | No |
| Auth handling | Complex | Token-based | Limited |
| Cost | High (browser instances) | Low | Low |
| Anti-bot risk | Medium-high | None | Low |

## Real-World Examples

### Web Research Agent
```
1. navigate_to_url("https://example.com/pricing")
2. extract_text("div.pricing-table")
3. take_screenshot()  -- for visual verification
4. Return pricing data to user
```

### Form Automation Agent
```
1. navigate_to_url("https://app.example.com/login")
2. fill_field("#email", "user@example.com")
3. fill_field("#password", "***")
4. click_element("button[type='submit']")
5. extract_text(".dashboard-content")
```

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| Selector breaks | Website UI redesign | Tool fails silently or selects wrong element | Use robust selectors (data-testid, aria-label) |
| Anti-bot block | Cloudflare, reCAPTCHA | Navigation fails | Stealth plugins, proxy rotation |
| Session timeout | Long-running agent loses session | Auth state lost, need re-login | Session persistence, cookie storage |
| Memory leak | Browser accumulates tabs/memory | Agent process OOM | Close pages aggressively, restart browser |
| Dynamic content miss | Content loads after JS execution | Extract returns empty | wait_for_selector before extracting |
| Infinite page load | Heavy page never reaches ready state | Timeout | Use domcontentloaded instead of load event |

## Source(s) and Further Reading

- Playwright Documentation -- browser automation API
- Puppeteer Documentation -- alternative browser automation
- "Browser Use" (GitHub) -- open-source agent browser automation
- Anthropic, "Computer Use" (2024) -- screenshot-based agent interaction
- "AgentQL" -- AI-powered web element selection
