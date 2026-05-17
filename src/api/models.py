"""Pydantic v2 request and response models for the Travel Planner API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class TravelRequestBody(BaseModel):
    """POST /plan request body."""

    destination: str = Field(..., min_length=1, max_length=200, description="Travel destination city or region")
    start_date: date = Field(..., description="Trip start date (YYYY-MM-DD)")
    end_date: date = Field(..., description="Trip end date (YYYY-MM-DD)")
    budget_min: float = Field(..., gt=0, description="Minimum budget")
    budget_max: float = Field(..., gt=0, description="Maximum budget")
    interests: list[str] = Field(..., min_length=1, max_length=10, description="1–10 interest keywords")
    num_travelers: int = Field(..., ge=1, le=20, description="Number of travelers (1–20)")

    @model_validator(mode="after")
    def _validate_dates_and_budget(self) -> "TravelRequestBody":
        today = date.today()
        if self.start_date < today:
            raise ValueError("start_date must be today or a future date")
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if self.budget_min >= self.budget_max:
            raise ValueError("budget_min must be less than budget_max")
        return self


class ReviewRequestBody(BaseModel):
    """POST /plan/{id}/review request body."""

    action: Literal["approve", "reject", "modify"] = Field(..., description="Review action")
    feedback: str | None = Field(None, max_length=2000, description="Optional feedback text")
    modifications: dict[str, Any] | None = Field(None, description="Optional modification details")


class PlanCreatedResponse(BaseModel):
    """Returned by POST /plan on success."""

    session_id: str
    status: str
    draft_itinerary: dict[str, Any] | None = None
    message: str | None = None


class PlanStatusResponse(BaseModel):
    """Returned by GET /plan/{id}."""

    session_id: str
    status: str
    workflow_stage: str
    hitl_status: str | None = None
    draft_itinerary: dict[str, Any] | None = None
    error: str | None = None
    revision_count: int = 0


class ReviewResponse(BaseModel):
    """Returned by POST /plan/{id}/review."""

    session_id: str
    status: str
    workflow_stage: str
    hitl_status: str | None = None
    draft_itinerary: dict[str, Any] | None = None
    final_plan: dict[str, Any] | None = None
    message: str | None = None


class FinalPlanResponse(BaseModel):
    """Returned by GET /plan/{id}/final."""

    session_id: str
    status: str
    final_plan: dict[str, Any]


class ErrorResponse(BaseModel):
    """Generic error response."""

    detail: str


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = "ok"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
