<p align="center">
  <h1 align="center">🛡️ AgentGuard</h1>
  <p align="center"><strong>The governance layer that lets companies trust their AI agents enough to actually deploy them.</strong></p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <a href="https://pypi.org/project/agentaudit-sdk/"><img src="https://img.shields.io/pypi/v/agentaudit-sdk.svg" alt="PyPI"></a>
  <a href="https://github.com/AviraL0013/agentguard/actions/workflows/tests.yml"><img src="https://github.com/AviraL0013/agentguard/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
</p>

---

> **62% of production AI teams** plan to improve observability in the next year. **Over 40% of agentic AI projects** will be canceled by 2027 due to inadequate risk controls. **Humans still verify 69%** of AI decisions because there are no guardrails they trust.
>
> AgentGuard fixes this. One SDK. Full audit trail. Every LLM call and tool use — intercepted, policy-checked, cost-tracked, and logged. **3 lines of code.**

---

## Quick Start

```python
from openai import OpenAI
from agentguard import AgentGuard

client = OpenAI()

guard = AgentGuard(
    policies=["pii", "content_filter", "cost_limit"],
    audit_path="audit.jsonl",
    cost_limit=5.00,
)

safe_client = guard.wrap_openai(client)

# Use exactly like the original — now with full protection
response = safe_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

Every call is now:
- ✅ **PII-scanned** — blocks emails, SSNs, credit cards, phone numbers  
- ✅ **Policy-checked** — blocks prompt injections, enforces budget limits  
- ✅ **Cost-tracked** — per-model, per-run, and daily spend tracking  
- ✅ **Audit-logged** — immutable JSON-lines trail for compliance  

## Installation

```bash
# Core (OpenAI support included)
pip install agentaudit-sdk

# With Anthropic support
pip install "agentaudit-sdk[anthropics]"
```

---

## 🤖 Anthropic Claude Integration

Wrap Claude exactly like OpenAI — **3 lines, full protection**.

```python
import anthropic
from agentguard import AgentGuard

client = anthropic.Anthropic()

guard = AgentGuard(
    policies=["pii", "content_filter", "cost_limit"],
    audit_path="audit.jsonl",
    cost_limit=5.00,
)

safe = guard.wrap_anthropic(client)

# Use exactly like the original — now fully protected
response = safe.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello, Claude!"}],
)
print(response.content[0].text)
```

Every call is now:
- ✅ **PII-scanned** — both `messages` list AND the top-level `system` prompt
- ✅ **Policy-checked** — prompt injections blocked, budget enforced
- ✅ **Cost-tracked** — accurate per-model pricing for all Claude 3 variants
- ✅ **Audit-logged** — immutable JSON-lines trail

### Async Claude

```python
import anthropic
from agentguard import AgentGuard

async with AgentGuard(policies=["pii", "content_filter"]) as guard:
    client = anthropic.AsyncAnthropic()
    safe = guard.wrap_anthropic_async(client)

    response = await safe.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=512,
        system="You are a helpful assistant.",   # 🛡️ system prompt is PII-scanned too
        messages=[{"role": "user", "content": "Summarise this report."}],
    )
    print(response.content[0].text)
```

### Supported Claude Models (with built-in pricing)

| Model | Input / 1M tokens | Output / 1M tokens |
|---|---|---|
| `claude-3-5-sonnet-20241022` | $3.00 | $15.00 |
| `claude-3-5-haiku-20241022` | $0.80 | $4.00 |
| `claude-3-opus-20240229` | $15.00 | $75.00 |
| `claude-3-sonnet-20240229` | $3.00 | $15.00 |
| `claude-3-haiku-20240307` | $0.25 | $1.25 |

---

## Features

### 🛡️ Built-in Policies

| Policy | What It Does |
|--------|-------------|
| `pii` | Blocks PII (emails, SSN, credit cards, phones, IPs) in inputs & outputs |
| `content_filter` | Blocks prompt injection attempts & system prompt extraction |
| `cost_limit` | Enforces per-run, daily, and total budget limits |
| `rate_limit` | Throttles calls per time window (sliding window) |
| `tool_restriction` | Blocklist/allowlist for agent tool usage |

### 🔧 Tool Guarding

Wrap any function — sync or async. Policies are enforced *before* the tool runs.

```python
def delete_database(db_name: str) -> str:
    ...

safe_delete = guard.wrap_tool(delete_database)
safe_delete(db_name="production")  # 🛡️ Blocked by tool_restriction policy
```

```python
# PII is caught in tool arguments too
def send_email(to: str, body: str) -> str:
    ...

safe_send = guard.wrap_tool(send_email)
safe_send(to="john@example.com", body="Hi")  # 🛡️ Blocked: PII detected
```

### ⚡ Full Async Support

Works with `AsyncOpenAI` and async tool functions — zero changes to your logic.

```python
from openai import AsyncOpenAI

async with AgentGuard(policies=["pii", "content_filter"]) as guard:
    client = AsyncOpenAI()
    safe = guard.wrap_openai_async(client)

    response = await safe.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    # Async tools — auto-detected
    async def fetch_data(url: str) -> str:
        ...

    safe_fetch = guard.wrap_tool(fetch_data)  # auto-detects async
    result = await safe_fetch(url="https://api.example.com")
```

### 💰 Cost Tracking

Real-time spend tracking with per-model pricing for GPT-4o, GPT-4o-mini, Claude, Gemini, o1, o3-mini, and more.

```python
report = guard.get_report()
# {
#     'total_cost_usd': 0.0234,
#     'total_tokens_in': 1500,
#     'total_tokens_out': 800,
#     'daily_cost_usd': 0.0234,
#     'run_cost_usd': 0.0120,
#     'policies_active': ['pii', 'content_filter', 'cost_limit']
# }
```

### 🎬 Audit Reader & Replay

The **killer feature**. Prove exactly what your agent did, step by step.

#### Python API

```python
from agentguard import AuditReader

reader = AuditReader("audit.jsonl")
run = reader.get_run("run_abc123")
run.print_trace()
```

```
╔══════════════════════════════════════════════════════════╗
║  AGENTGUARD RUN TRACE                                   ║
╠══════════════════════════════════════════════════════════╣
║  Run ID:     run_abc123                                  ║
║  Events:     3                                           ║
║  Tokens:     1,500 in / 800 out                          ║
║  Cost:       $0.0234                                     ║
║  Violations: None                                        ║
╠══════════════════════════════════════════════════════════╣
║  Step 1  LLM  OK
║    Model:  gpt-4o
║    user: "Process the customer refund"
║    assistant: "I'll process that refund now."
║
║  Step 2  TOOL  OK
║    Tool:     process_refund
║    Args:     {"order_id": "ORD-12345", "amount": 49.99}
║    Duration: 230ms
║
║  Step 3  LLM  OK
║    Model:  gpt-4o
║    assistant: "The refund of $49.99 has been processed."
╚══════════════════════════════════════════════════════════╝
```

#### CLI Tool

```bash
# List all runs with summary stats
agentguard --file audit.jsonl runs

# Step-by-step replay of any run
agentguard --file audit.jsonl replay <run_id>
agentguard --file audit.jsonl replay <run_id> --delay 0.5  # slow replay

# Show all policy violations (audit-ready)
agentguard --file audit.jsonl violations

# Dashboard — costs, models, tools, violations
agentguard --file audit.jsonl stats

# Search events by any content
agentguard --file audit.jsonl search "delete_database"

# Export for compliance reports
agentguard --file audit.jsonl export --format json -o report.json
agentguard --file audit.jsonl export --format csv -o audit.csv

# Live tail — watch events in real-time
agentguard --file audit.jsonl tail
```

### 🔌 Custom Policies

Build your own — just subclass `Policy` and implement `evaluate()`.

```python
from agentguard import Policy, PolicyResult, PolicyAction
from agentguard.core.events import LLMCallEvent

class NoProfanityPolicy(Policy):
    name = "no_profanity"
    supported_events = [LLMCallEvent]

    def evaluate(self, event):
        bad_words = ["damn", "hell"]
        content = str(event.messages).lower()
        if any(w in content for w in bad_words):
            return PolicyResult(
                action=PolicyAction.BLOCK,
                policy_name=self.name,
                reason="Profanity detected",
            )
        return PolicyResult(action=PolicyAction.ALLOW, policy_name=self.name)

guard = AgentGuard(policies=[NoProfanityPolicy(), "pii"])
```

### 🔔 Human-in-the-Loop Escalation

```python
def on_escalation(event):
    print(f"ALERT: {event.reason}")
    # Send to Slack, PagerDuty, email, etc.

guard = AgentGuard(
    policies=["pii", "content_filter"],
    on_escalation=on_escalation,  # supports async callbacks too
)
```

---

## Run the Demo

```bash
python examples/basic_usage.py
```

## Run Tests

```bash
pip install agentaudit-sdk[dev]
pytest tests/ -v
# 104 tests passing in <1 second
```

---

## Architecture

```
Your App → AI Agent → 🛡️ AgentGuard SDK → Tool / LLM API
                            │
                     ┌──────┴──────────┐
                     │   Interceptor   │  ← before/after hooks
                     ├─────────────────┤
                     │  Policy Engine  │  ← PII, Cost, Content, Rate, Tool
                     ├─────────────────┤
                     │  PII Detector   │  ← Regex (pluggable to ML/Presidio)
                     ├─────────────────┤
                     │  Cost Tracker   │  ← Per-model pricing, run/daily/total
                     ├─────────────────┤
                     │  Audit Logger   │  ← Thread-safe, JSON-lines, rotation
                     ├─────────────────┤
                     │  Audit Reader   │  ← Query, filter, replay, CLI
                     └─────────────────┘
```

```
src/agentguard/
├── core/
│   ├── events.py          # Pydantic event models (run_id grouping)
│   ├── interceptor.py     # Central before/after hooks
│   └── guard.py           # Main orchestrator (3-line API)
├── policies/
│   ├── base.py            # Policy engine + event-type filtering
│   ├── pii_policy.py      # PII blocking
│   ├── cost_policy.py     # Budget enforcement
│   ├── tool_policy.py     # Tool blocklist/allowlist
│   ├── rate_limit_policy.py  # Sliding window rate limiter
│   └── content_policy.py  # Prompt injection detection
├── detectors/
│   └── pii.py             # Regex PII detector (pluggable Protocol)
├── tracking/
│   └── cost.py            # Token & cost tracking
├── logging/
│   ├── audit.py           # Thread-safe JSON-lines logger
│   └── reader.py          # Audit reader + replay engine
├── integrations/
│   └── openai.py          # Sync + Async OpenAI proxy
└── cli.py                 # CLI audit reader (7 commands)
```

## Why AgentGuard?

| Problem | How AgentGuard Solves It |
|---------|------------------------|
| "Nobody knows what our agent is doing" | Every LLM call and tool use is logged with full context |
| "We can't trace agent failures" | Run-level audit trails with step-by-step replay |
| "Auditors want proof" | JSON-lines logs + CSV export mapped to compliance frameworks |
| "Humans verify 69% of AI decisions" | Policy guardrails let you reduce human review confidently |
| "Agents keep leaking PII" | Automatic PII detection and blocking on all inputs & outputs |
| "AI costs are unpredictable" | Per-run, daily, and total budget limits with real-time tracking |
| "Demo works, production doesn't" | The missing operating system — cost controls, guardrails, audit trails |

---

## License

MIT
