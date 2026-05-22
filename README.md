# route-llm

Multi-provider LLM proxy with time-based routing, auto-failover, and usage tracking.

Acts as a unified API gateway that accepts **OpenAI-compatible** and **Anthropic** requests, then routes them to the best available backend provider based on weight, time rules, and failure state.

## Features

- **Dual protocol** — Exposes both `/v1/chat/completions` (OpenAI) and `/v1/messages` (Anthropic) endpoints
- **Weighted routing** — Select provider by weight; higher-weight providers are preferred
- **Time-based rules** — Per-provider rules to redirect or remap models during specific time windows (e.g., redirect to cheaper model during peak hours)
- **Cross-provider redirect** — Rules can redirect to a different provider entirely
- **Auto-failover** — Retries up to 3 times on HTTP 429/5xx, falling back to alternative providers
- **Failure cooldown** — Failed providers are skipped for 30 minutes (configurable) to avoid repeated failures
- **Usage tracking** — Records token usage per provider/model to daily JSONL files; queryable via API
- **Prompt caching metrics** — Tracks cache read/creation tokens for Anthropic providers
- **Streaming** — SSE streaming support for Anthropic endpoint
- **Auth** — Optional API key validation via `x-api-key` header or `Authorization: Bearer`
- **Docker support** — Multi-arch build (amd64/arm64) with docker-compose

## Quick Start

### 1. Create config

```bash
cp config.example.toml config.toml
# Edit config.toml with your API keys
```

### 2. Run

```bash
# Local
uv run uvicorn route_llm.main:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker compose up -d
```

### 3. Use

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Configuration

Configuration is via TOML file. By default, the config is loaded from (in order):

1. `$ROUTE_LLM_CONFIG` environment variable
2. `./config.toml`
3. `~/.config/route-llm/config.toml`
4. `/etc/route-llm/config.toml`

### Structure

```toml
api_key = "your-api-key"       # API key for client auth (optional)

[providers]

# OpenAI-compatible providers
[providers.openai]
[providers.openai.<vendor_name>]
api_key = "YOUR_API_KEY"
base_url = "https://api.example.com/v1"
weight = 5                    # Higher = preferred

[providers.openai.<vendor_name>.models."model-name"]
level = "large"               # "small" | "large"
max_tokens = 65536

# Anthropic-format providers
[providers.anthropic]
[providers.anthropic.<vendor_name>]
api_key = "YOUR_API_KEY"
base_url = "https://api.example.com/anthropic"
weight = 5

[providers.anthropic.<vendor_name>.models."model-name"]
level = "large"
max_tokens = 131072
```

### Routing Rules

Per-provider rules allow time-based model redirection:

```toml
[[providers.anthropic.zhipu.rules]]
name = "peak-hour-downgrade"
time_range = { start = "14:00", end = "18:00" }
match_model = "glm-5.1,glm-5-turbo"    # Match incoming models (comma-separated)
model = "glm-4.7"                       # Override to this model
priority = 10                           # Higher priority wins

# Cross-provider redirect (leave empty to redirect within same provider)
# provider = "anthropic.baidu"          # Redirect to different provider
```

Rules support cross-midnight time ranges (e.g., `start = "22:00", end = "06:00"`).

### Environment Variable Overrides

API keys and base URLs can be overridden via environment variables:

```bash
ROUTE_LLM_PROVIDER_OPENAI_SILICONFLOW_API_KEY=sk-xxx
ROUTE_LLM_PROVIDER_ANTHROPIC_ZHIPU_BASE_URL=https://custom-url/api/anthropic
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completion |
| `/v1/messages` | POST | Anthropic-format messages (supports streaming) |
| `/v1/usage` | GET | Query usage records |

### Usage Query

```bash
curl "http://localhost:8000/v1/usage?start_date=2025-01-01&end_date=2025-01-31&provider=zhipu&model=glm"
```

Query params (all optional):
- `start_date` — ISO date (default: today)
- `end_date` — ISO date (default: today)
- `provider` — substring filter on provider name
- `model` — substring filter on model name

## Architecture

```
Client ──► FastAPI ──► Router ──► Provider ──► Upstream API
                           │
                    FailureTracker    UsageTracker
                    (30min cooldown)   (JSONL files)
```

The **Router** selects the best provider by:
1. Finding candidates that have the requested model
2. Picking the highest-weight provider (skipping blocked/excluded)
3. Applying provider-level rules (time-based model override/redirect)

If the upstream returns a retryable error (429, 500, 502, 503), the request falls back to the next-best provider. Failed providers enter a cooldown (default 30 min).

## Development

```bash
# Install
uv sync

# Run tests
uv run pytest

# Run with hot reload
uv run python -m route_llm
```

## Build & Push Docker Image

```bash
make build   # Build for local architecture
make push    # Multi-arch build (linux/amd64,linux/arm64) and push
```

## License

MIT