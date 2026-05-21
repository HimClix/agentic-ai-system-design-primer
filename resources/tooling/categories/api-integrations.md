# API Integration Tools for Agents
> Every SaaS API you integrate is a dependency you maintain forever. Rate limits, auth rotation, schema versioning, and circuit breakers are not optional -- they are the cost of admission.

## What It Is

API integration tools connect agents to external SaaS services -- Stripe for payments, GitHub for code, Slack for messaging, Jira for tickets, and hundreds of others. Each integration wraps a third-party API into a tool the LLM can call, handling authentication, rate limits, error mapping, and response formatting.

The key challenge: each API has different auth mechanisms, rate limit strategies, error formats, pagination patterns, and versioning schemes. A production agent with 5 SaaS integrations must handle 5 different failure modes, 5 auth systems, and 5 rate limit policies simultaneously.

## How It Works

### Architecture per Integration

```
LLM calls tool (e.g., create_github_issue)
  |
  v
[Tool Layer] -- validates input, maps to API call
  |
  v
[Auth Layer] -- injects API key / OAuth token
  |              |-- Key rotation (if near expiry)
  |              |-- Token refresh (OAuth)
  |              |-- Vault fetch (secrets management)
  |
  v
[Rate Limit Layer] -- pre-flight check
  |                    |-- Under limit? -> proceed
  |                    |-- Over limit? -> wait (Retry-After) or reject
  |
  v
[Circuit Breaker] -- per-integration breaker
  |
  v
[HTTP Client] -- actual API call with timeout
  |
  v
[Response Mapper] -- API-specific response -> standard ToolResult
  |                   |-- Extract relevant fields
  |                   |-- Map errors to error taxonomy
  |                   |-- Truncate large responses
  |
  v
Return structured result to LLM
```

## Production Implementation

### Base Integration Class

```python
"""Base class for SaaS API integrations in agent tools."""
import httpx
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration for an API integration."""
    name: str                          # e.g., "stripe", "github"
    base_url: str                      # e.g., "https://api.stripe.com/v1"
    auth_type: str = "bearer"          # "bearer", "basic", "api_key_header"
    auth_header: str = "Authorization" # Header name for auth
    api_key_prefix: str = "Bearer"     # e.g., "Bearer", "token", "Bot"
    timeout_seconds: int = 15
    max_retries: int = 3
    rate_limit_rpm: int = 100          # Requests per minute
    api_version: Optional[str] = None  # e.g., "2024-06-20" for Stripe


class RateLimiter:
    """Simple sliding window rate limiter per integration."""
    
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []
    
    def check(self) -> tuple[bool, Optional[float]]:
        """Check if request is allowed. Returns (allowed, wait_seconds)."""
        now = time.time()
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        
        if len(self._timestamps) >= self.max_requests:
            wait = self._timestamps[0] - cutoff
            return False, wait
        
        self._timestamps.append(now)
        return True, None


class BaseAPIIntegration(ABC):
    """Base class for all API integrations."""
    
    def __init__(self, config: APIConfig, api_key: str):
        self.config = config
        self._api_key = api_key
        self._rate_limiter = RateLimiter(config.rate_limit_rpm)
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            headers=self._build_headers(),
        )
    
    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.auth_type == "bearer":
            headers[self.config.auth_header] = f"{self.config.api_key_prefix} {self._api_key}"
        elif self.config.auth_type == "api_key_header":
            headers[self.config.auth_header] = self._api_key
        if self.config.api_version:
            headers["Stripe-Version"] = self.config.api_version  # Example
        return headers
    
    async def _request(
        self, method: str, path: str, 
        params: dict = None, json_body: dict = None,
    ) -> dict:
        """Make an authenticated, rate-limited API request."""
        
        # Rate limit check
        allowed, wait_time = self._rate_limiter.check()
        if not allowed:
            return {
                "success": False,
                "error_type": "rate_limit",
                "message": f"Rate limit reached for {self.config.name}. "
                           f"Wait {wait_time:.0f} seconds.",
                "retry_after_seconds": int(wait_time) + 1,
            }
        
        try:
            response = await self._client.request(
                method=method,
                url=path,
                params=params,
                json=json_body,
            )
            
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", "5"))
                return {
                    "success": False,
                    "error_type": "rate_limit",
                    "message": f"{self.config.name} API rate limited. "
                               f"Wait {retry_after}s.",
                    "retry_after_seconds": int(retry_after),
                }
            
            if response.status_code >= 400:
                return self._map_error(response)
            
            return {
                "success": True,
                "data": response.json(),
            }
        
        except httpx.TimeoutException:
            return {
                "success": False,
                "error_type": "timeout",
                "message": f"{self.config.name} API timed out after "
                           f"{self.config.timeout_seconds}s.",
            }
        except httpx.ConnectError:
            return {
                "success": False,
                "error_type": "transient",
                "message": f"Cannot connect to {self.config.name} API.",
            }
    
    def _map_error(self, response: httpx.Response) -> dict:
        """Map API-specific errors to standard error taxonomy."""
        status = response.status_code
        try:
            body = response.json()
        except Exception:
            body = {"message": response.text[:500]}
        
        error_message = self._extract_error_message(body)
        
        if status in {400, 422}:
            return {
                "success": False,
                "error_type": "validation",
                "message": f"{self.config.name}: {error_message}",
                "field_errors": self._extract_field_errors(body),
            }
        if status == 401:
            return {
                "success": False,
                "error_type": "auth",
                "message": f"{self.config.name} authentication failed.",
            }
        if status == 404:
            return {
                "success": False,
                "error_type": "not_found",
                "message": f"Resource not found in {self.config.name}.",
            }
        return {
            "success": False,
            "error_type": "internal",
            "message": f"{self.config.name} returned HTTP {status}.",
        }
    
    @abstractmethod
    def _extract_error_message(self, body: dict) -> str:
        """Extract human-readable error from API-specific response."""
        pass
    
    @abstractmethod
    def _extract_field_errors(self, body: dict) -> dict:
        """Extract field-level errors from API-specific response."""
        pass


# --- Concrete Integration: GitHub ---

class GitHubIntegration(BaseAPIIntegration):
    """GitHub API integration for agent tools."""
    
    def __init__(self, token: str):
        super().__init__(
            config=APIConfig(
                name="github",
                base_url="https://api.github.com",
                auth_type="bearer",
                api_key_prefix="token",
                timeout_seconds=15,
                rate_limit_rpm=60,  # GitHub: 5000/hr for auth, ~83/min
            ),
            api_key=token,
        )
    
    def _extract_error_message(self, body: dict) -> str:
        return body.get("message", "Unknown GitHub error")
    
    def _extract_field_errors(self, body: dict) -> dict:
        errors = body.get("errors", [])
        return {
            e.get("field", "unknown"): e.get("message", "invalid")
            for e in errors if isinstance(e, dict)
        }
    
    # --- Tools ---
    
    async def create_issue(self, owner: str, repo: str, title: str, 
                           body: str = "", labels: list[str] = None) -> dict:
        """Create a GitHub issue in a repository."""
        payload = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        
        result = await self._request("POST", f"/repos/{owner}/{repo}/issues", json_body=payload)
        
        if result["success"]:
            issue = result["data"]
            return {
                "success": True,
                "data": {
                    "issue_number": issue["number"],
                    "url": issue["html_url"],
                    "title": issue["title"],
                    "state": issue["state"],
                },
            }
        return result
    
    async def list_issues(self, owner: str, repo: str, state: str = "open",
                          labels: str = None, per_page: int = 10) -> dict:
        """List issues in a repository."""
        params = {"state": state, "per_page": min(per_page, 100)}
        if labels:
            params["labels"] = labels
        
        result = await self._request("GET", f"/repos/{owner}/{repo}/issues", params=params)
        
        if result["success"]:
            issues = result["data"]
            return {
                "success": True,
                "data": {
                    "issues": [
                        {
                            "number": i["number"],
                            "title": i["title"],
                            "state": i["state"],
                            "author": i["user"]["login"],
                            "labels": [l["name"] for l in i.get("labels", [])],
                            "url": i["html_url"],
                        }
                        for i in issues[:per_page]
                    ],
                    "count": len(issues),
                },
            }
        return result
```

## Decision Tree: Adding a New API Integration

```
Do you need to integrate with a SaaS API?
  |
  YES --> Does an official SDK/toolkit exist for agents?
  |         |
  |         YES --> Use it (Stripe Agent Toolkit, GitHub MCP, etc.)
  |         NO  --> Build a custom integration using BaseAPIIntegration
  |
  v
For the custom integration:
  |
  |-- Step 1: Find rate limits in API docs
  |-- Step 2: Identify auth mechanism (API key, OAuth, etc.)
  |-- Step 3: Map API error format to error taxonomy
  |-- Step 4: Define tools (one per action, single responsibility)
  |-- Step 5: Add circuit breaker config per integration
  |-- Step 6: Write schema descriptions with API-specific examples
```

## When NOT to Build Custom Integrations

- **MCP server exists**: Check the MCP registry first -- many SaaS APIs have pre-built MCP servers
- **Agent toolkit exists**: Stripe, GitHub, and others have official agent toolkits
- **Rarely used**: If the integration is called less than once per 100 agent sessions, consider manual delegation
- **API is unstable**: Beta APIs with frequent breaking changes will require constant maintenance

## Tradeoffs

| Aspect | Custom Integration | Official SDK/Toolkit | MCP Server |
|--------|-------------------|---------------------|------------|
| Control | Full | Moderate | Limited |
| Maintenance | You maintain | Vendor maintains | Community maintains |
| Schema quality | Your responsibility | Usually good | Varies |
| Rate limit handling | Your implementation | Usually built-in | Varies |
| Auth management | Your implementation | Usually built-in | Built-in |
| Setup time | Days | Hours | Minutes |
| Customization | Unlimited | Limited to SDK | Plugin architecture |

## Real-World Examples

### Multi-Integration Agent: Customer Support
```
Integration 1: Stripe (payment lookup, refunds)
  - Rate limit: 100 req/s per mode
  - Auth: API key with test/live modes
  - Circuit breaker: 5 failures in 60s

Integration 2: Zendesk (ticket management)
  - Rate limit: 700 req/min
  - Auth: OAuth 2.0 with token refresh
  - Circuit breaker: 3 failures in 30s

Integration 3: Slack (notifications)
  - Rate limit: 1 req/s per channel
  - Auth: Bot token
  - Circuit breaker: 5 failures in 120s (Slack is reliable)

Each integration has its own:
  - Circuit breaker instance
  - Rate limiter instance
  - Auth token management
  - Error mapping
```

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| API version deprecation | SaaS vendor deprecates old API | Tools start failing | Pin API version in headers, monitor deprecation notices |
| Rate limit exhaustion | Multiple agents share same API key | 429 errors for all agents | Per-agent rate limits, key pooling |
| Auth token expiry | OAuth token expires mid-session | 401 errors | Refresh before expiry, retry on 401 with fresh token |
| Schema drift | API adds/removes fields | Response parsing breaks | Version pin, defensive parsing |
| Cascading failure | One integration's failure triggers others | Agent unable to function | Per-integration circuit breakers |
| Secret exposure | API key leaked in logs/errors | Account compromise | Never log auth headers, use vault |

## Source(s) and Further Reading

- Stripe Agent Toolkit (GitHub) -- production API integration for agents
- MCP (Model Context Protocol) Specification -- standardized tool protocol
- Anthropic MCP Server Registry -- pre-built integrations
- "9 Practices for Production Agentic Workflows" (arXiv, 2025) -- per-integration reliability
- Stripe API Rate Limiting -- industry-standard rate limit design
