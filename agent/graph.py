"""Wires the nodes into a LangGraph state machine."""
from langgraph.graph import END, START, StateGraph

from . import nodes
from .modes import MODES
from .state import ResearchState


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("recall", nodes.recall)
    g.add_node("plan", nodes.plan)
    g.add_node("act", nodes.act)
    g.add_node("tools", nodes.run_tools)
    g.add_node("record", nodes.record)
    g.add_node("reflect", nodes.reflect)
    g.add_node("write", nodes.write)
    g.add_node("remember", nodes.remember)

    g.add_edge(START, "recall")
    g.add_edge("recall", "plan")
    g.add_edge("plan", "act")
    g.add_conditional_edges("act", nodes.after_act, {"tools": "tools", "record": "record"})
    g.add_edge("tools", "act")  # observe -> reason again
    g.add_conditional_edges("record", nodes.after_record, {"act": "act", "reflect": "reflect"})
    g.add_conditional_edges("reflect", nodes.after_reflect, {"act": "act", "write": "write"})
    g.add_edge("write", "remember")
    g.add_edge("remember", END)
    return g.compile()


graph = build_graph()


MAX_GOAL_CHARS = 500


def validate_goal(goal: str) -> str:
    """Shared input check for the UI, CLI and API: a blank or huge goal would
    waste the free-tier token quota on a run that can't be useful."""
    goal = (goal or "").strip()
    if not goal:
        raise ValueError("Please enter a research goal.")
    if len(goal) > MAX_GOAL_CHARS:
        raise ValueError(f"The research goal is too long ({len(goal)} characters, max {MAX_GOAL_CHARS}).")
    return goal


def run(goal: str, mode: str = "general", on_step=None) -> dict:
    """Run the agent end-to-end. `on_step(node_name, new_trace_lines)` is
    called after every node so UIs can show live progress."""
    goal = validate_goal(goal)
    if mode not in MODES:
        raise ValueError(f"Unknown research mode {mode!r}. Choose one of: {', '.join(MODES)}.")
    state: dict = {"goal": goal, "mode": mode, "trace": []}
    # recursion_limit bounds total node executions as a final safety net.
    for chunk in graph.stream(state, stream_mode="updates", config={"recursion_limit": 100}):
        for node_name, update in chunk.items():
            seen = len(state["trace"])
            state.update(update or {})
            if on_step:
                on_step(node_name, state["trace"][seen:])
    return state
