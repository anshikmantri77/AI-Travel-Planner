# Architecture — AI Travel Planner

## System Overview

The AI Travel Planner is a multi-agent system orchestrated by **LangGraph** and served via
**FastAPI**. Two specialised LLM agents — a Research Agent and an Itinerary Planner Agent —
collaborate through a stateful workflow that includes a human-in-the-loop (HITL) approval
gate. State is persisted across HTTP request boundaries using LangGraph's `MemorySaver`
checkpointer, allowing the workflow to pause at the HITL step and resume when the user
submits feedback.

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                          │
│                                                                     │
│  POST /plan ──► create session ──► invoke graph (run to interrupt)  │
│  GET  /plan/{id} ──► read session state                            │
│  POST /plan/{id}/review ──► resume graph with Command(resume=...)  │
│  GET  /plan/{id}/final ──► return final_plan or 409                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                              │
│                                                                     │
│  ┌──────────────┐    ┌──────────┐    ┌────────────────┐            │
│  │   validate    │──►│ research  │──►│ plan_itinerary  │            │
│  │   _request    │   │  (Agent1) │   │   (Agent2)      │            │
│  └──────────────┘    └──────────┘    └───────┬────────┘            │
│                                              │                      │
│                                              ▼                      │
│                                   ┌──────────────────┐             │
│                                   │ hitl_checkpoint   │             │
│                                   │  interrupt()      │             │
│                                   └────────┬─────────┘             │
│                                            │                        │
│                            Command(resume=feedback)                 │
│                                            │                        │
│                                            ▼                        │
│                                   ┌──────────────────┐             │
│                                   │ process_feedback  │             │
│                                   └──┬─────┬─────┬───┘             │
│                                      │     │     │                  │
│                          approve     │ reject   modify              │
│                                      │     │     │                  │
│                                      ▼     │     │                  │
│                              ┌───────────┐ │     │                  │
│                              │ finalize   │ │     │                  │
│                              └───────────┘ │     │                  │
│                                      ▲     │     │                  │
│                                      │     ▼     ▼                  │
│                                      │  research / plan_itinerary   │
│                                      │     (revision loop,          │
│                                      │      max 3 cycles)           │
│                                      └──────────────────────────────│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
User Request
     │
     ▼
validate_request ──► Checks dates, budget, interests
     │
     ▼
Research Agent
  ├── Tool: web_search  (Serper API — attractions, tips, safety)
  └── Tool: get_weather (Open-Meteo — 7-day forecast)
     │
     ▼  research_output (JSON)
Planner Agent
  ├── Tool: allocate_budget  (heuristic split by style)
  └── Tool: score_activities (keyword-match ranking)
     │
     ▼  draft_itinerary (JSON)
HITL Checkpoint ──► interrupt()  ──► HTTP 200 with draft
     │
     │  (user reviews via POST /plan/{id}/review)
     ▼
process_feedback
  ├── approve  ──► finalize ──► final_plan
  ├── reject   ──► research (re-run both agents)
  └── modify   ──► plan_itinerary (re-run planner only)
```

---

## State Schema

| Field              | Type                                         | Purpose                              |
|--------------------|----------------------------------------------|--------------------------------------|
| `session_id`       | `str`                                        | Unique plan identifier               |
| `travel_request`   | `dict`                                       | Original user input                  |
| `research_output`  | `dict`                                       | Research Agent structured output      |
| `draft_itinerary`  | `dict`                                       | Planner Agent structured output       |
| `hitl_status`      | `pending│approved│rejected│modified`         | Current review state                 |
| `hitl_feedback`    | `str`                                        | User's textual feedback              |
| `hitl_modifications` | `dict`                                     | Specific changes requested           |
| `final_plan`       | `dict`                                       | Approved and assembled plan          |
| `workflow_stage`   | `str`                                        | Current graph node                   |
| `error`            | `str │ None`                                 | Error message if failed              |
| `revision_count`   | `int`                                        | Tracks reject/modify cycles (max 3)  |

---

## Key Design Decisions

### LangGraph `interrupt()` for HITL

LangGraph's first-class `interrupt()` primitive pauses graph execution and serialises
the full state to the checkpointer. When the user submits feedback via the review
endpoint, the API calls `graph.invoke(Command(resume=feedback), config)` which
picks up exactly where execution stopped. This is cleaner than manual state-machine
code and guarantees consistency.

### `MemorySaver` Checkpointer

For a take-home assignment, the in-memory `MemorySaver` avoids external infrastructure
(Redis, Postgres). Each HTTP request supplies a `thread_id` config so LangGraph can
look up the correct checkpoint. In production this would be swapped for a persistent
store.

### `Command(goto=...)` for Dynamic Routing

The `process_feedback` node returns a `Command` object with `goto` set to the
appropriate next node. This avoids complex conditional edges and keeps routing logic
co-located with the feedback-processing code.

### Agentic Tool Loop

Both agents use a manual invoke → check-tool-calls → execute → feed-back loop
(max 10 iterations) rather than LangChain's `AgentExecutor`. This gives full control
over error handling and iteration limits while keeping the code transparent.

---

## Agent Detail

### Research Agent

- **LLM**: Configurable — `ChatAnthropic` (default) or `ChatOpenAI`
- **Tools**: `search_web`, `get_weather_forecast`
- **Output**: Structured JSON with destination overview, attractions, tips, safety,
  weather, accommodation areas, cuisine highlights
- **Fallback**: If no API keys are set, tools return mock data with warnings

### Planner Agent

- **LLM**: Same configurable LLM
- **Tools**: `allocate_budget`, `score_activities`
- **Output**: Day-by-day itinerary with activities, costs, accommodation,
  packing suggestions
- **Budget logic**: Pure-Python heuristic split by travel style
  (budget / mid-range / luxury)

---

## API Endpoints

| Method | Path               | Description                        | Key Status Codes |
|--------|--------------------|------------------------------------|------------------|
| GET    | `/health`          | Liveness check                     | 200              |
| POST   | `/plan`            | Submit travel request              | 200, 422         |
| GET    | `/plan/{id}`       | Poll plan status / draft           | 200, 404         |
| POST   | `/plan/{id}/review`| Submit HITL feedback               | 200, 404, 409    |
| GET    | `/plan/{id}/final` | Retrieve finalised plan            | 200, 404, 409    |
