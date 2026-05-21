# Multi-Region Deployment for Agent Systems

> Deploying agent systems across regions for latency reduction, data residency compliance, and provider proximity. Where to deploy, how to replicate state, and the cost implications.

## What It Is

Multi-region deployment distributes agent infrastructure across geographic regions to achieve:

1. **LLM provider proximity**: Deploy your agent backend in the same region as Anthropic/OpenAI endpoints (~50ms saved per call)
2. **User proximity**: Reduce round-trip latency for end users
3. **Data residency**: Keep EU user data in EU datacenters (GDPR)
4. **High availability**: Survive regional outages

### LLM Provider Regions (2025)

| Provider | Primary Region | Other Regions | API Endpoint Latency |
|----------|---------------|---------------|---------------------|
| **Anthropic** | us-east-1 (Virginia) | eu-west-1 (via AWS Bedrock) | 50-100ms from us-east-1 |
| **OpenAI** | us-east (Azure East US) | eu-west, asia | 50-150ms from us-east |
| **Google (Gemini)** | us-central1 | Multiple (via Vertex AI) | 50-100ms |
| **AWS Bedrock** | us-east-1, us-west-2, eu-west-1, ap-southeast-1 | Multiple | 30-80ms same region |

### Impact of Region Choice on Latency

```
Scenario: Agent in ap-south-1 (Mumbai) calling Anthropic API in us-east-1

Network latency: Mumbai → Virginia = ~200ms round trip
LLM processing:  ~1,000ms (Claude Sonnet)
Total per call:   ~1,200ms

Agent in us-east-1 calling Anthropic API in us-east-1:
Network latency: same region = ~5ms round trip
LLM processing:  ~1,000ms
Total per call:   ~1,005ms

Savings per call: ~195ms
For 5-step agent task: ~975ms total savings (5 × 195ms)
```

## How It Works

### Multi-Region Architecture

```
                    ┌─────────────────────┐
                    │   Global Load        │
                    │   Balancer           │
                    │   (Route53/CloudFlare)│
                    └───┬─────────┬───────┘
                        │         │
              ┌─────────▼──┐  ┌──▼─────────┐
              │ US-EAST-1   │  │ EU-WEST-1   │
              │             │  │             │
              │ ┌─────────┐ │  │ ┌─────────┐ │
              │ │ Agent    │ │  │ │ Agent    │ │
              │ │ Backend  │ │  │ │ Backend  │ │
              │ └────┬────┘ │  │ └────┬────┘ │
              │      │      │  │      │      │
              │ ┌────▼────┐ │  │ ┌────▼────┐ │
              │ │ LLM     │ │  │ │ LLM     │ │
              │ │ Gateway  │ │  │ │ Gateway  │ │
              │ │ (LiteLLM)│ │  │ │ (LiteLLM)│ │
              │ └────┬────┘ │  │ └────┬────┘ │
              │      │      │  │      │      │
              │ ┌────▼────┐ │  │ ┌────▼────┐ │
              │ │Anthropic │ │  │ │ Bedrock  │ │
              │ │ Direct   │ │  │ │ Claude   │ │
              │ │ (5ms)    │ │  │ │ (30ms)   │ │
              │ └─────────┘ │  │ └─────────┘ │
              │             │  │             │
              │ ┌─────────┐ │  │ ┌─────────┐ │
              │ │ Memory   │ │  │ │ Memory   │ │
              │ │ (Primary)│◄├──┤►│ (Replica)│ │
              │ └─────────┘ │  │ └─────────┘ │
              └─────────────┘  └─────────────┘
                                     │
                              ┌──────▼──────┐
                              │ AP-SOUTH-1   │
                              │ (Mumbai)     │
                              │              │
                              │ Agent Backend│
                              │ + Bedrock    │
                              │ + Memory     │
                              │   (Replica)  │
                              └──────────────┘
```

### Data Residency Routing

```
EU User Request:
  1. DNS resolves to EU-WEST-1 endpoint
  2. Agent backend in EU-WEST-1 processes request
  3. LLM call → AWS Bedrock EU (Claude in eu-west-1)
  4. Memory stored in EU Postgres/vector DB
  5. Response served from EU

  EU data NEVER leaves EU datacenters ✓

US User Request:
  1. DNS resolves to US-EAST-1 endpoint
  2. Agent backend in US-EAST-1 processes request
  3. LLM call → Anthropic API direct (us-east-1)
  4. Memory stored in US Postgres
  5. Response served from US
```

## Production Implementation

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Region(Enum):
    US_EAST_1 = "us-east-1"
    EU_WEST_1 = "eu-west-1"
    AP_SOUTH_1 = "ap-south-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"


@dataclass
class RegionConfig:
    region: Region
    llm_provider: str            # "anthropic_direct" | "bedrock" | "openai"
    llm_endpoint: str
    memory_host: str
    memory_is_primary: bool
    tool_endpoints: dict[str, str]


# --- Region Configuration ---
REGION_CONFIGS = {
    Region.US_EAST_1: RegionConfig(
        region=Region.US_EAST_1,
        llm_provider="anthropic_direct",
        llm_endpoint="https://api.anthropic.com",
        memory_host="memory-us-east-1.internal:5432",
        memory_is_primary=True,
        tool_endpoints={
            "search": "https://search-us.internal",
            "database": "https://db-us.internal",
        },
    ),
    Region.EU_WEST_1: RegionConfig(
        region=Region.EU_WEST_1,
        llm_provider="bedrock",
        llm_endpoint="bedrock.eu-west-1.amazonaws.com",
        memory_host="memory-eu-west-1.internal:5432",
        memory_is_primary=False,
        tool_endpoints={
            "search": "https://search-eu.internal",
            "database": "https://db-eu.internal",
        },
    ),
    Region.AP_SOUTH_1: RegionConfig(
        region=Region.AP_SOUTH_1,
        llm_provider="bedrock",
        llm_endpoint="bedrock.ap-south-1.amazonaws.com",
        memory_host="memory-ap-south-1.internal:5432",
        memory_is_primary=False,
        tool_endpoints={
            "search": "https://search-ap.internal",
            "database": "https://db-ap.internal",
        },
    ),
}


class MultiRegionRouter:
    """
    Routes agent requests to the appropriate region based on:
    1. User's geographic location (latency)
    2. Data residency requirements (compliance)
    3. Provider availability (failover)
    """

    # GDPR: EU users must be served from EU
    EU_COUNTRIES = {
        "DE", "FR", "IT", "ES", "NL", "BE", "AT", "PL",
        "SE", "DK", "FI", "IE", "PT", "CZ", "RO", "HU",
        "GR", "BG", "HR", "SK", "LT", "SI", "LV", "EE",
        "CY", "LU", "MT",
    }

    def route_request(
        self,
        user_country: str,
        data_residency_required: bool = False,
        preferred_region: Optional[Region] = None,
    ) -> RegionConfig:
        """Determine the best region for this request."""

        # Rule 1: Data residency overrides everything
        if data_residency_required and user_country in self.EU_COUNTRIES:
            return REGION_CONFIGS[Region.EU_WEST_1]

        # Rule 2: User preference (if valid)
        if preferred_region and preferred_region in REGION_CONFIGS:
            return REGION_CONFIGS[preferred_region]

        # Rule 3: Geographic proximity
        if user_country in self.EU_COUNTRIES:
            return REGION_CONFIGS[Region.EU_WEST_1]
        elif user_country in {"IN", "PK", "BD", "LK", "NP"}:
            return REGION_CONFIGS[Region.AP_SOUTH_1]
        else:
            return REGION_CONFIGS[Region.US_EAST_1]

    def failover(self, failed_region: Region) -> RegionConfig:
        """Return the next best region if the primary fails."""
        failover_map = {
            Region.US_EAST_1: Region.EU_WEST_1,
            Region.EU_WEST_1: Region.US_EAST_1,
            Region.AP_SOUTH_1: Region.US_EAST_1,
        }
        fallback = failover_map.get(failed_region, Region.US_EAST_1)
        return REGION_CONFIGS[fallback]


# --- Cross-Region Memory Replication ---

class CrossRegionMemorySync:
    """
    Handles memory replication across regions.
    
    Strategy: Primary-replica with async replication.
    - Writes go to primary (us-east-1)
    - Reads go to local replica (eventually consistent)
    - Critical reads go to primary (strong consistency)
    """

    def __init__(self, local_region: Region):
        self.local = REGION_CONFIGS[local_region]
        self.primary = REGION_CONFIGS[Region.US_EAST_1]

    async def write_memory(self, user_id: str, memory: dict) -> None:
        """
        Write to primary, async replicate to local.
        
        For EU data residency: primary IS the EU region.
        """
        if self.local.memory_is_primary:
            # Write locally (this IS the primary)
            await self._write_to_db(self.local.memory_host, user_id, memory)
        else:
            # Write to primary, local replica will catch up via replication
            await self._write_to_db(self.primary.memory_host, user_id, memory)

    async def read_memory(
        self,
        user_id: str,
        strong_consistency: bool = False,
    ) -> list[dict]:
        """
        Read from local replica (fast) or primary (consistent).
        
        Use strong_consistency=True for critical reads where
        stale data could cause incorrect agent behavior.
        """
        if strong_consistency or self.local.memory_is_primary:
            return await self._read_from_db(self.primary.memory_host, user_id)
        else:
            return await self._read_from_db(self.local.memory_host, user_id)

    async def _write_to_db(self, host: str, user_id: str, memory: dict):
        """Write to a specific database host."""
        # Implementation with asyncpg
        pass

    async def _read_from_db(self, host: str, user_id: str) -> list[dict]:
        """Read from a specific database host."""
        pass
```

## Decision Tree: Multi-Region Deployment

```
    Do I need multi-region?
                │
     ┌──────────▼──────────┐
     │ Do you have users    │
     │ in multiple          │
     │ continents?          │
     └──┬──────────────┬──┘
       Yes             No
        │               │
   ┌────▼───────────┐  Single region
   │ Do you have    │  (deploy in
   │ data residency │   us-east-1 for
   │ requirements?  │   best LLM latency)
   │ (GDPR, etc.)   │
   └──┬────────┬───┘
    Yes       No
     │         │
  REQUIRED   ┌──▼────────────┐
  (EU data   │ Is latency     │
   in EU)    │ critical?      │
             │ (need <500ms   │
             │  response)     │
             └──┬────────┬───┘
               Yes       No
                │         │
           RECOMMENDED  Optional
                          (single region
                           + CDN may suffice)
```

## When NOT to Use

1. **Single-market applications**: If all users are in one country, multi-region adds complexity without benefit.
2. **Internal tools**: Internal agent tools used by employees in one office do not need multi-region.
3. **Early-stage startups**: Multi-region is expensive and complex. Start single-region, add regions when you have product-market fit.
4. **Batch processing**: If the agent processes data in batches (not real-time), deploy wherever compute is cheapest.
5. **When using a single LLM provider**: If you only use Anthropic Direct API (us-east-1), deploying your backend elsewhere adds latency.

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Lower latency for global users | 2-3x infrastructure cost |
| GDPR/data residency compliance | Cross-region state consistency is hard |
| Provider failover capability | Operational complexity (3x monitoring) |
| Regional outage resilience | Different LLM availability per region |
| Deploy close to LLM providers | Memory replication lag (eventual consistency) |

## Cost Implications

```
Single region (us-east-1):
  Compute: $2,000/month (3x t3.xlarge)
  Database: $500/month (RDS PostgreSQL)
  Vector DB: $300/month (Qdrant)
  Total: ~$2,800/month

Multi-region (us-east-1 + eu-west-1 + ap-south-1):
  Compute: $6,000/month (3x in each region)
  Database: $1,800/month (primary + 2 read replicas)
  Vector DB: $900/month (3 Qdrant instances)
  Cross-region data transfer: $200-500/month
  Total: ~$8,900-9,200/month

  Premium: ~3.3x single-region cost
```

## Failure Modes

### 1. Replication Lag
User writes memory in EU, immediately reads in US. The data has not replicated yet.
**Mitigation**: Read-after-write consistency for the same user. Route user to the same region within a session.

### 2. Split Brain
Network partition between regions. Both think they are primary and accept writes.
**Mitigation**: Use managed databases (RDS, Aurora Global) that handle split-brain. Implement leader election.

### 3. Provider Availability Mismatch
Claude is available in us-east-1 but experiencing issues in eu-west-1 (Bedrock). Failover sends EU traffic to US, violating data residency.
**Mitigation**: Maintain per-region provider health checks. Fail over to different providers in the same region before cross-region failover.

### 4. Cost Overrun
Cross-region data transfer costs accumulate faster than expected.
**Mitigation**: Minimize cross-region replication. Cache aggressively in each region. Monitor data transfer costs daily.

### 5. Configuration Drift
Regions get out of sync (different agent versions, different prompt templates).
**Mitigation**: GitOps with automated deployment to all regions. Configuration parity checks.

## Sources and Further Reading

- [AWS Bedrock Regions](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html)
- [Anthropic API Status](https://status.anthropic.com/)
- [GDPR Data Residency Requirements](https://gdpr.eu/article-44-transfer-of-personal-data/)
- [Aurora Global Databases](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/aurora-global-database.html)
- [CloudFlare Geo-Routing](https://developers.cloudflare.com/load-balancing/understand-basics/traffic-steering/)
