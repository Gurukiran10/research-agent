"""Minimal REST API so the agent can be called from other systems.

    uvicorn api:app --reload
    POST /research  {"goal": "...", "mode": "general|competitor|market|leads"}
    POST /feedback  {"run_id": 1, "goal": "...", "rating": 1, "feedback": "..."}
"""
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from agent import memory
from agent.graph import run
from agent.modes import MODES
from agent.nodes import learn_from_feedback

app = FastAPI(title="Research Agent API")


class ResearchRequest(BaseModel):
    goal: str
    mode: Literal["general", "competitor", "market", "leads"] = "general"


class FeedbackRequest(BaseModel):
    run_id: int
    goal: str
    rating: int  # 1 = helpful, -1 = not helpful
    feedback: str = ""


@app.post("/research")
def research(req: ResearchRequest):
    state = run(req.goal, mode=req.mode)
    return {
        "run_id": state["run_id"],
        "plan": state["plan"],
        "findings": state["findings"],
        "report": state["report"],
        "trace": state["trace"],
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    return {"lessons_learned": learn_from_feedback(req.run_id, req.goal, req.rating, req.feedback)}


@app.get("/modes")
def get_modes():
    return {k: {"label": m.label, "description": m.description, "example": m.example} for k, m in MODES.items()}


@app.get("/memory")
def get_memory():
    return {"lessons": memory.get_lessons(), "runs": memory.list_runs()}
