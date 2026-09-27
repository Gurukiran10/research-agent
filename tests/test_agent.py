"""Offline tests - no API key or internet needed.

    python -m pytest -q      (or)      python tests/test_agent.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage  # noqa: E402

from agent import memory, nodes  # noqa: E402
from agent.tools import calculator  # noqa: E402


def test_calculator_is_safe_and_correct():
    assert calculator.invoke({"expression": "(120-80)/80*100"}) == "50.0"
    assert calculator.invoke({"expression": "(801.8/698.63)^(1/5)-1"}) == "0.027931"  # ^ means power
    assert "error" in calculator.invoke({"expression": "__import__('os')"}).lower()


def test_memory_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "m.sqlite")
    run_id = memory.save_run("vector databases comparison", "Report text")
    memory.rate_run(run_id, 1, "great")
    memory.add_lesson("Include a comparison table.")
    assert memory.get_lessons() == ["Include a comparison table."]
    assert memory.find_related_reports("best vector databases")[0]["id"] == run_id

    # full run details round-trip so the UI can reopen a past run
    detailed = memory.save_run("g2", "R2", "market", {"plan": ["q1"], "trace": ["PLAN: q1"]})
    assert memory.get_run(detailed)["details"] == {"plan": ["q1"], "trace": ["PLAN: q1"]}
    assert memory.get_run(run_id)["details"] == {}  # saved without details
    assert memory.count_runs() == 2 and len(memory.list_runs(None)) == 2
    memory.clear_all()
    assert memory.count_runs() == 0 and memory.get_lessons() == []


class FakeLLM:
    """Scripted stand-in for ChatGroq that walks the graph through one full
    plan -> tool call -> answer -> reflect(gap) -> follow-up -> write cycle."""

    def __init__(self):
        self.structured = None
        self.tool_turn = {}

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, **kwargs):
        fake = FakeLLM()
        fake.structured = schema
        fake.tool_turn = self.tool_turn
        return fake

    def invoke(self, messages):
        if self.structured is nodes.Plan:
            return nodes.Plan(sub_questions=["What is X?", "Why does X matter?"])
        if self.structured is nodes.Critique:
            FakeLLM.reflections = getattr(FakeLLM, "reflections", 0) + 1
            if FakeLLM.reflections == 1:
                return nodes.Critique(sufficient=False, critique="Missing numbers", follow_up_questions=["How big is X?"])
            return nodes.Critique(sufficient=True, critique="Good")
        if isinstance(messages, str):  # writer prompt
            return AIMessage(content="# Report\nTL;DR done")
        question = messages[1].content
        if not self.tool_turn.get(question):
            self.tool_turn[question] = True
            return AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "2+2"}, "id": "c1"}])
        return AIMessage(content=f"Answer to {question} (source: https://example.com/x)")


def test_full_workflow_with_fake_llm(tmp_path, monkeypatch):
    from agent import graph as graph_mod

    fake = FakeLLM()
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: fake)
    monkeypatch.setattr(nodes, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "m.sqlite")
    FakeLLM.reflections = 0

    state = graph_mod.run("Research X", mode="market")

    assert state["plan"] == ["What is X?", "Why does X matter?", "How big is X?"]  # reflection added a step
    assert len(state["findings"]) == 3
    # the fake LLM "cites" a URL no tool ever returned -> it must be dropped
    assert state["findings"][0]["sources"] == []
    assert state["report"].startswith("# Report")
    assert "## Sources" in state["report"]
    assert Path(state["report_path"]).exists()
    assert any(t.startswith("REFLECT: gaps found") for t in state["trace"])
    assert any("calculator" in t for t in state["trace"])
    assert memory.list_runs()[0]["goal"] == "Research X"
    assert memory.list_runs()[0]["mode"] == "market"
    saved = memory.get_run(state["run_id"])["details"]
    assert saved["plan"] == state["plan"] and len(saved["findings"]) == 3 and saved["trace"]
    assert "-market-" in Path(state["report_path"]).name
    assert "mode=Market Research" in state["trace"][0]


def test_writer_sees_which_source_backs_each_fact(tmp_path, monkeypatch):
    seen = {}

    class CapturingLLM:
        def invoke(self, prompt):
            seen["prompt"] = prompt
            return AIMessage(content="# R")

    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: CapturingLLM())
    monkeypatch.setattr(nodes, "REPORTS_DIR", tmp_path)
    finding = {
        "question": "q",
        "answer": "Salesforce costs $25 (source: https://a.com/x). Pipedrive costs $15 (source: https://b.com/y/).",
        "sources": ["https://a.com/x", "https://b.com/y"],
    }
    out = nodes.write({"goal": "g", "mode": "competitor", "findings": [finding], "trace": []})
    assert "Salesforce costs $25 (source: [1])" in seen["prompt"]
    assert "Pipedrive costs $15 (source: [2])" in seen["prompt"]
    assert "https://" not in seen["prompt"]  # URLs never reach the writer
    assert out["report"].endswith("1. https://a.com/x\n2. https://b.com/y")


def test_writer_survives_all_models_rate_limited(tmp_path, monkeypatch):
    """If every model is out of quota at the last step, the run must still
    deliver its verified findings instead of crashing."""
    from agent import llm

    monkeypatch.setattr(llm, "PATIENCE_S", 0)

    class RateLimited:
        def invoke(self, prompt):
            raise RuntimeError("429 rate limit")

    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: RateLimited())
    monkeypatch.setattr(nodes, "REPORTS_DIR", tmp_path)
    finding = {"question": "What is X?", "answer": "X is Y (source: https://a.com/x).", "sources": ["https://a.com/x"]}
    out = nodes.write({"goal": "g", "findings": [finding], "critique": "thin evidence", "trace": []})
    assert "## Q1. What is X?" in out["report"] and "X is Y (source: [1])" in out["report"]
    assert "## Limitations" in out["report"] and out["report"].endswith("1. https://a.com/x")
    assert "report assembled from verified findings" in out["trace"][-1]


def test_fallback_chain_has_three_distinct_models():
    from agent import llm

    names = [name for name, _ in llm.LLM(0.2).models]
    assert len(names) == len(set(names)) >= 3


def test_daily_limit_skips_model_but_chain_still_answers(monkeypatch):
    from agent import llm

    class Model:
        def __init__(self, error=None):
            self.error, self.calls = error, 0

        def invoke(self, prompt):
            self.calls += 1
            if self.error:
                raise RuntimeError(self.error)
            return "ok"

    monkeypatch.setattr(llm, "_exhausted_until", {})
    spent = Model("Error code: 429 - Rate limit reached ... tokens per day (TPD): Limit 200000")
    backup = Model()
    chain = llm._Chain([("big", spent), ("backup", backup)])
    assert chain.invoke("q") == "ok" and chain.invoke("q") == "ok"
    assert spent.calls == 1 and backup.calls == 2  # exhausted model not retried on the 2nd call


def test_entry_points_compile():
    import py_compile

    root = Path(__file__).resolve().parent.parent
    for name in ("app.py", "cli.py", "api.py"):
        py_compile.compile(str(root / name), doraise=True)


def test_every_mode_fills_every_prompt():
    """A typo in a mode or a prompt placeholder would only show up at runtime -
    format every prompt with every mode to catch it offline."""
    from agent import prompts
    from agent.modes import MODES

    for mode in MODES.values():
        assert mode.sections and mode.sections[-1] == "Limitations", mode.key
        prompts.PLANNER.format(goal="g", max_q=3, today="2026-09-27", lessons="-", related="-",
                               mode_label=mode.label, mode_planning=mode.planning)
        prompts.EXECUTOR.format(goal="g", today="2026-09-27", step=1, total=3, question="q", previous="-", budget=3,
                                lessons="-", mode_label=mode.label, mode_execution=mode.execution)
        prompts.REFLECTOR.format(goal="g", findings="-", today="2026-09-27", mode_label=mode.label,
                                 mode_sections=", ".join(mode.sections))
        prompts.WRITER.format(goal="g", findings="-", today="2026-09-27", critique="-", lessons="-",
                              mode_label=mode.label, mode_report=mode.report)


def test_invented_tool_falls_back_to_answer(monkeypatch):
    """gpt-oss sometimes calls tools it was trained on but we don't provide
    (e.g. 'web_read'); the agent must answer from evidence instead of crashing."""

    class HallucinatingLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            if isinstance(messages, str):
                return AIMessage(content="Answer from evidence")
            return AIMessage(content="", tool_calls=[{"name": "web_read", "args": {"id": 6}, "id": "x"}])

    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: HallucinatingLLM())
    state = {"goal": "g", "plan": ["q"], "step_index": 0, "messages": [], "trace": []}
    out = nodes.act(state)
    assert out["messages"][-1].content == "Answer from evidence"
    assert not out["messages"][-1].tool_calls


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
