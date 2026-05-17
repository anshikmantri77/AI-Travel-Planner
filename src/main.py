"""FastAPI application entry point for the AI Travel Planner."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router
from src.api.session_store import SessionStore
from src.orchestrator import compile_graph
from src.utils.helpers import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialise graph + session store on startup."""
    setup_logging()

    # Compile the LangGraph workflow with MemorySaver checkpointer
    app.state.graph = compile_graph()

    # Create the in-memory session store
    app.state.sessions = SessionStore()

    yield

    # Cleanup (nothing persistent to tear down with in-memory store)


app = FastAPI(
    title="AI Travel Planner",
    description=(
        "Multi-agent travel planning system with LangGraph orchestration "
        "and human-in-the-loop approval."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
