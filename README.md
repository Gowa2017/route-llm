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
- **Peak / off-peak pricing** — Per-vendor peak windows (time of day + weekday) with optional peak prices; usage is costed per record timestamp
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
input_price = 2.0             # per million tokens, off-peak base price
cache_read_price = 0.4
output_price = 8.0
peak_input_price = 4.0        # optional; falls back to the base price per field
peak_cache_read_price = 0.8
peak_output_price = 16.0
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

### Peak / Off-Peak Pricing

`input_price` / `cache_read_price` / `output_price` are **off-peak base prices**. A vendor
may declare `peak_windows`; usage recorded inside a window is billed at the matching
`peak_*_price` (any `peak_*_price` left unset falls back to the base price).

```toml
[providers.anthropic.zhipu]
peak_windows = [
  { days = ["mon", "tue", "wed", "thu", "fri"], start = "08:00", end = "12:00" },
  { days = ["mon", "tue", "wed", "thu", "fri"], start = "14:00", end = "18:00" },
  { start = "20:00", end = "22:00" },                       # days omitted = every day
  { days = ["fri"], start = "23:00", end = "02:00" },       # cross-midnight
]

[providers.anthropic.zhipu.models."glm-4.7"]
input_price = 2.0
output_price = 8.0
peak_input_price = 4.0
peak_output_price = 16.0
# peak_windows = []     # model-level override; [] disables peak pricing for this model
```

- `days` accepts `mon`..`sun` (case-insensitive); omitted means every day.
- `start > end` crosses midnight; the after-midnight part belongs to the window's start day
  (the example above covers Friday 23:00 → Saturday 02:00).
- Model-level `peak_windows` overrides the vendor default; `[]` disables it.
- Classification uses each usage record's own `timestamp`, so historical data is re-costed
  correctly. Records with a missing, date-only, or unparseable timestamp count as off-peak.
- Providers without `peak_windows`, and models without any `peak_*_price`, are reported as a
  single row exactly as before.

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
| `/v1/usage/table` | GET | Same data, rendered as a text table |

### Usage Query

```bash
curl "http://localhost:8000/v1/usage?start_date=2025-01-01&end_date=2025-01-31&provider=zhipu&model=glm"
```

Query params (all optional):
- `start_date` — ISO date (default: today)
- `end_date` — ISO date (default: today)
- `provider` — substring filter on provider name
- `model` — substring filter on model name

Each returned row carries the aggregated tokens, latency stats and cost. When a model has peak
pricing configured it is split into one row per bucket, each with a `bucket` field (`"peak"` or
`"offpeak"`); the table endpoint appends ` (高峰)` / ` (非高峰)` to those rows and lists the
off-peak row first.

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