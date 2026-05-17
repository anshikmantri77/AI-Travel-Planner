# AI Travel Planner

A multi-agent travel planning system built with **LangGraph**, **FastAPI**, and LLM-powered
agents. Users submit a travel request; two specialised agents research the destination and
build a day-by-day itinerary; the system then pauses for human-in-the-loop approval before
producing the final plan.

The project demonstrates stateful workflow orchestration with LangGraph's `interrupt()` /
`Command(resume=...)` primitives, tool-augmented LLM agents, and a clean REST API layer.
State is persisted across HTTP requests via LangGraph's checkpointer so the workflow can
pause at the review step and resume asynchronously.

---

## Architecture

```
User ──► POST /plan ──► Orchestrator
                            │
                  ┌─────────┴──────────┐
                  ▼                    ▼
           Research Agent        Planner Agent
           (web_search,         (budget_allocator,
            weather)             activity_scorer)
                  │                    │
                  └─────────┬──────────┘
                            ▼
                     HITL Checkpoint
                      (interrupt)
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
       approve           reject            modify
          │                 │                 │
          ▼                 ▼                 ▼
       Finalize      Research Agent     Planner Agent
          │              (loop)            (loop)
          ▼
   GET /plan/{id}/final
```

Detailed architecture documentation is in [`docs/architecture.md`](docs/architecture.md).

---

## Setup

### Prerequisites

- Python 3.11+
- API keys (optional — the system falls back to mock data if keys are missing)

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/ai-travel-planner.git
cd ai-travel-planner

# Copy env template and fill in your keys
cp .env.example .env
# Edit .env with your GROQ_API_KEY, SERPER_API_KEY, etc.

# Install dependencies
make install
# or: pip install -r requirements.txt
```

### Environment Variables

| Variable           | Required | Default                  | Description                     |
|--------------------|----------|--------------------------|---------------------------------|
| `GROQ_API_KEY`| Yes*     | —                             | GROQ API key for GROQ  |
| `OPENAI_API_KEY`   | No       | —                        | Alternative LLM provider        |
| `SERPER_API_KEY`   | Yes       | —                        | Serper web search (mock if absent) |
| `LLM_PROVIDER`     | No       | `groq`                   | `groq` or `openai`         |
| `LLM_MODEL`        | No       | `llama-3.3-70b-versatile`| Model name                     |
| `LOG_LEVEL`        | No       | `INFO`                   | Logging level                   |

\* Required unless `LLM_PROVIDER=groq` and `GROQ_API_KEY` is set.

---

## Running the Application

```bash
make run
# or: uvicorn src.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `/docs`.

---

## Running Tests

```bash
make test
# or: pytest tests/ -v
```

Tests mock external services (LLM calls, HTTP APIs) so no API keys are needed.

---

## Example API Requests

### 1. Submit a Travel Request

```bash
curl -s -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Tokyo, Japan",
    "start_date": "2026-09-01",
    "end_date": "2026-09-05",
    "budget_min": 800,
    "budget_max": 2000,
    "interests": ["temples", "street food", "anime", "nature"],
    "num_travelers": 2
  }' | python -m json.tool
```

**Response (200):**
```json
{
  "session_id": "abc123-...",
  "status": "awaiting_review",
  "draft_itinerary": {
    "trip_summary": "5-day Tokyo adventure...",
    "days": [ ... ]
  }
}
```

### 2. Check Plan Status

```bash
curl -s http://localhost:8000/plan/abc123-... | python -m json.tool
```

**Response (200):**
```json
{
  "session_id": "abc123-...",
  "status": "awaiting_review",
  "workflow_stage": "hitl_checkpoint",
  "draft_itinerary": { ... },
  "hitl_status": "pending",
  "error": null
}
```

### 3. Approve the Plan

```bash
curl -s -X POST http://localhost:8000/plan/abc123-.../review \
  -H "Content-Type: application/json" \
  -d '{
    "action": "approve",
    "feedback": "Looks great!"
  }' | python -m json.tool
```

**Response (200):**
```json
{
  "session_id": "abc123-...",
  "status": "approved",
  "workflow_stage": "finalize"
}
```

### 4. Reject with Feedback

```bash
curl -s -X POST http://localhost:8000/plan/abc123-.../review \
  -H "Content-Type: application/json" \
  -d '{
    "action": "reject",
    "feedback": "I want more focus on street food and less on museums."
  }' | python -m json.tool
```

### 5. Modify Specific Parts

```bash
curl -s -X POST http://localhost:8000/plan/abc123-.../review \
  -H "Content-Type: application/json" \
  -d '{
    "action": "modify",
    "feedback": "Swap day 3 afternoon activity",
    "modifications": {
      "day_3": {"afternoon": {"activity": "Akihabara electronics tour"}}
    }
  }' | python -m json.tool
```

### 6. Retrieve Final Plan

```bash
curl -s http://localhost:8000/plan/abc123-.../final | python -m json.tool
```

**Response (200 after approval):**
```json
{
  "session_id": "abc123-...",
  "status": "approved",
  "final_plan": {
    "trip_summary": "...",
    "days": [ ... ],
    "total_budget_used": 1850,
    "research_highlights": { ... }
  }
}
```

**Response (409 before approval):**
```json
{
  "detail": "Plan not yet approved. Current status: pending"
}
```

---

## Design Decisions

### Why LangGraph?

LangGraph provides first-class support for stateful, interruptible workflows. Its
`interrupt()` / `Command(resume=...)` API maps directly to the HITL requirement:
the graph pauses execution, serialises state to a checkpointer, and resumes from
the exact same point when feedback arrives. This is far cleaner than manually
saving/restoring state in the application layer.

### MemorySaver vs Redis

`MemorySaver` (in-memory checkpointer) was chosen to keep the project zero-dependency
and runnable with `pip install` alone. It stores LangGraph checkpoints in a Python
dict, which is sufficient for development and evaluation. In production, this would
be replaced by `RedisSaver` or `PostgresSaver` for durability and horizontal scaling.

### HITL State Persistence

The workflow pauses at `hitl_checkpoint` via `interrupt()`. The session store maps
each `session_id` to a LangGraph `thread_id`. When the review endpoint is called,
the API resumes the graph by passing the same `thread_id` in the config — LangGraph
loads the checkpoint and continues. No manual serialisation is needed.

### Agentic Tool Loop (not AgentExecutor)

Both agents use a manual loop: invoke LLM → check for tool calls → execute tools →
feed results back → repeat. This gives explicit control over iteration limits (max 10),
error handling, and output parsing, while keeping the code easy to follow and debug.

### Dynamic Routing with `Command(goto=...)`

The `process_feedback` node returns a `Command` with `goto` pointing to the next
node (`finalize`, `research`, or `plan_itinerary`). This co-locates routing logic
with feedback processing and avoids a tangle of conditional edges.

---

## Tradeoffs

| Decision                         | Benefit                        | Cost                                  |
|----------------------------------|--------------------------------|---------------------------------------|
| In-memory session store          | Zero infrastructure setup      | State lost on restart; single process |
| In-memory checkpointer           | No Redis/Postgres dependency   | Not production-durable                |
| Mock fallback for tools          | Works without API keys         | Less realistic research output        |
| Keyword-based activity scoring   | Fast, no external calls        | No semantic understanding             |
| Hardcoded budget percentages     | Deterministic, testable        | Doesn't learn from real price data    |
| Single-file agents (no streaming)| Simpler code, easier testing   | No real-time progress updates         |

---

## What I Would Improve with More Time

1. **Redis/Postgres checkpointer** — Replace `MemorySaver` with `RedisSaver` for
   durable state that survives restarts and supports multiple workers.
2. **Streaming (SSE)** — Stream agent progress to the client in real time using
   Server-Sent Events so users see research happening live.
3. **Authentication & rate limiting** — Add JWT-based auth and per-user rate limits.
4. **Docker Compose** — Containerise the app with Redis, add a `docker-compose.yml`.
5. **Richer tools** — Integrate real hotel/flight APIs (Amadeus, Skyscanner), a
   currency converter, and a distance/travel-time calculator.
6. **Semantic activity scoring** — Use embeddings instead of keyword matching for
   more nuanced interest-to-activity matching.
7. **Observability** — Add OpenTelemetry tracing to track agent execution, tool
   latency, and LLM token usage.
8. **Frontend** — A lightweight React UI showing the workflow stages and enabling
   drag-and-drop itinerary editing.
9. **Caching** — Cache research results for popular destinations to reduce LLM
   calls and API costs.
10. **Comprehensive error recovery** — Retry transient API failures, circuit-breaker
    for external services.

---

## Assumptions

- The evaluator has Python 3.11+ installed.
- `MemorySaver` (in-memory) is acceptable for a take-home; production would use a
  persistent store.
- Mock tool responses are acceptable when API keys are not provided.
- The LLM is assumed to return well-structured JSON when prompted; light parsing
  fallbacks are included but not exhaustive.
- A single-process, single-worker deployment is sufficient for demonstration.
- Session TTL of 1 hour is reasonable for evaluation purposes.

---

## License

This project is a take-home assignment submission and is not licensed for redistribution.
