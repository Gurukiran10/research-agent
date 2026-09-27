"""Minimal REST API so the agent can be called from other systems.

    uvicorn api:app --reload
    POST /research  {"goal": "...", "mode": "general|competitor|market|leads"}
    POST /feedback  {"run_id": 1, "goal": "...", "rating": 1, "feedback": "..."}
"""
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from agent import memory
from agent.graph import MAX_GOAL_CHARS, run
from agent.modes import MODES
from agent.nodes import learn_from_feedback

app = FastAPI(title="Research Agent API")


class ResearchRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=MAX_GOAL_CHARS)
    mode: Literal["general", "competitor", "market", "leads"] = "general"

    @field_validator("goal")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("goal must not be blank")
        return v.strip()


class FeedbackRequest(BaseModel):
    run_id: int
    goal: str
    rating: Literal[1, -1]  # 1 = helpful, -1 = not helpful
    feedback: str = Field(default="", max_length=1000)


@app.post("/research")
def research(req: ResearchRequest):
    try:
        state = run(req.goal, mode=req.mode)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if not state.get("is_research", True):  # small talk: answered directly, nothing saved
        return {"is_research": False, "reply": state["report"]}
    return {
        "is_research": True,
        "run_id": state["run_id"],
        "plan": state["plan"],
        "findings": state["findings"],
        "report": state["report"],
        "trace": state["trace"],
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if memory.get_run(req.run_id) is None:
        raise HTTPException(status_code=404, detail=f"No research run with id {req.run_id}.")
    return {"lessons_learned": learn_from_feedback(req.run_id, req.goal, req.rating, req.feedback)}


@app.get("/modes")
def get_modes():
    return {k: {"label": m.label, "description": m.description, "example": m.example} for k, m in MODES.items()}


@app.get("/memory")
def get_memory():
    return {"lessons": memory.get_lessons(), "runs": memory.list_runs()}


@app.get("/runs/{run_id}")
def get_run(run_id: int):
    saved = memory.get_run(run_id)
    if saved is None:
        raise HTTPException(status_code=404, detail=f"No research run with id {run_id}.")
    return saved
