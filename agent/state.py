"""The shared state that flows through every node of the graph.

Every node reads what it needs from this dict and returns only the keys it
changes; LangGraph merges the update back into the state. This is how the
agent keeps context across steps.
"""
from typing import TypedDict

from langchain_core.messages import BaseMessage


class Finding(TypedDict):
    question: str
    answer: str
    sources: list[str]


class ResearchState(TypedDict, total=False):
    # --- input ---
    goal: str
    mode: str                   # research mode key, see agent/modes.py

    # --- long-term memory pulled in at the start ---
    lessons: list[str]          # user-feedback lessons from earlier runs
    related_reports: list[str]  # short summaries of similar past research

    # --- planning ---
    plan: list[str]             # ordered sub-questions still to be answered
    step_index: int             # which sub-question the executor is on

    # --- execution (ReAct scratchpad for the *current* sub-question) ---
    messages: list[BaseMessage]  # replaced (not appended) by each node
    tool_calls_this_step: int

    # --- results ---
    findings: list[Finding]
    reflection_rounds: int
    critique: str

    # --- output ---
    report: str
    report_path: str
    run_id: int
    trace: list[str]            # human-readable log of every decision
