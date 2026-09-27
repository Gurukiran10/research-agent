"""Regression tests for bugs found in the QA pass (offline, no API key needed).

    python -m pytest -q
"""
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import memory, prompts, tools  # noqa: E402
from agent.graph import MAX_GOAL_CHARS, run, validate_goal  # noqa: E402

calc = lambda expr: tools.calculator.invoke({"expression": expr})  # noqa: E731
read = lambda url: tools.read_webpage.invoke({"url": url})  # noqa: E731


# ---------- calculator ----------
def test_calculator_rejects_huge_powers_instead_of_hanging():
    assert "exponent too large" in calc("9**9**9")


@pytest.mark.parametrize("expr, expected", [
    ("2^10", "1024"),                 # ^ means power, not XOR
    ("46.3%", "0.463"),               # percentages
    ("1,240/455", "2.725275"),        # thousands separators
    ("10 % 3", "1"),                  # modulo still works
    ("(1240/455)**(1/8)-1", "0.133512"),
])
def test_calculator_understands_how_llms_write_math(expr, expected):
    assert calc(expr) == expected


@pytest.mark.parametrize("expr", ["", "   ", "1/0", "__import__('os')", "open('x')"])
def test_calculator_errors_are_messages_not_crashes(expr):
    assert calc(expr).startswith("Calculation error")


# ---------- page reader: SSRF protection and content checks ----------
@pytest.mark.parametrize("url", [
    "http://localhost:8501", "http://127.0.0.1", "http://10.0.0.1", "http://192.168.1.1",
    "http://169.254.169.254/latest/meta-data",  # cloud metadata endpoint
    "file:///C:/Windows/win.ini", "ftp://example.com", "not a url",
])
def test_page_reader_refuses_local_and_non_http_urls(url):
    out = read(url)
    assert out.startswith("Could not read page") and ("not allowed" in out or "only public" in out
                                                       or "could not be resolved" in out)


def _fake_dns(ip):
    return lambda *a, **k: [(socket.AF_INET6 if ":" in ip else socket.AF_INET, 1, 6, "", (ip, 443))]


def test_nat64_address_wrapping_a_public_ip_is_allowed(monkeypatch):
    # mobile networks return 64:ff9b::/96 addresses; 64:ff9b::9765:32a wraps 151.101.3.42 (public)
    monkeypatch.setattr(tools.socket, "getaddrinfo", _fake_dns("64:ff9b::9765:32a"))
    assert tools._public_http_url("https://arxiv.org") is None


def test_nat64_address_wrapping_a_private_ip_is_blocked(monkeypatch):
    monkeypatch.setattr(tools.socket, "getaddrinfo", _fake_dns("64:ff9b::a00:1"))  # wraps 10.0.0.1
    assert "not allowed" in tools._public_http_url("https://sneaky.example")


# ---------- input validation ----------
@pytest.mark.parametrize("goal", ["", "   ", "\n\t", "x" * (MAX_GOAL_CHARS + 1)])
def test_blank_or_huge_goals_are_rejected_before_spending_tokens(goal):
    with pytest.raises(ValueError):
        validate_goal(goal)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown research mode"):
        run("a real goal", mode="nope")


def test_api_validates_input(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import api

    monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "m.sqlite")
    calls = []
    monkeypatch.setattr(api, "run", lambda goal, mode="general": calls.append(goal) or {
        "run_id": 1, "plan": [], "findings": [], "report": "r", "trace": []})
    c = TestClient(api.app)
    assert c.post("/research", json={"goal": ""}).status_code == 422
    assert c.post("/research", json={"goal": "   "}).status_code == 422
    assert c.post("/research", json={"goal": "x" * 5000}).status_code == 422
    assert c.post("/research", json={"goal": "x", "mode": "nope"}).status_code == 422
    assert c.post("/feedback", json={"run_id": 999, "goal": "g", "rating": 1}).status_code == 404
    assert c.post("/feedback", json={"run_id": 1, "goal": "g", "rating": 7}).status_code == 422
    assert c.get("/runs/999").status_code == 404
    assert calls == []  # invalid requests never reached the agent
    assert c.post("/research", json={"goal": "  ok goal  "}).status_code == 200
    assert calls == ["ok goal"]


# ---------- prompts ----------
def test_every_prompt_tells_the_model_todays_date():
    for name in ("PLANNER", "EXECUTOR", "REFLECTOR", "WRITER"):
        assert "{today}" in getattr(prompts, name), name


def test_critic_judges_by_evidence_not_training_cutoff():
    assert "knowledge cutoff" in prompts.REFLECTOR


# ---------- triage: small talk must not trigger a research run ----------
class _NoLLM:
    """Fails the test if any model is called."""

    def __getattr__(self, name):
        raise AssertionError("the LLM should not be called for this input")


@pytest.mark.parametrize("message", ["hii", "Hello!", "hey there", "thanks", "good morning", "ok"])
def test_greetings_get_an_instant_reply_without_llm_or_memory(message, tmp_path, monkeypatch):
    from agent import nodes

    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: _NoLLM())
    monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "m.sqlite")
    state = run(message)
    assert state["is_research"] is False and "research agent" in state["report"]
    assert memory.count_runs() == 0  # small talk is not saved as research history


@pytest.mark.parametrize("goal", ["history of India", "hi-tech industry in Bangalore", "testing frameworks for python"])
def test_topics_that_start_like_greetings_are_still_research(goal):
    from agent import nodes

    assert not nodes.GREETING_RE.match(goal)


def test_clear_requests_skip_the_llm_intent_check(monkeypatch):
    from agent import nodes

    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: _NoLLM())
    out = nodes.triage({"goal": "Compare LangGraph and CrewAI for agents", "trace": []})
    assert out["is_research"] is True


@pytest.mark.parametrize("is_research, expected_route", [(True, "recall"), (False, "end")])
def test_short_inputs_are_decided_by_the_llm(is_research, expected_route, monkeypatch):
    from agent import nodes

    class Decider:
        def with_structured_output(self, schema, **kw):
            return self

        def invoke(self, prompt):
            return nodes.Triage(is_research=is_research, reply="I research topics - try 'EV market in India'.")

    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: Decider())
    out = nodes.triage({"goal": "who are you", "trace": []})
    assert nodes.after_triage(out) == expected_route
    if not is_research:
        assert out["report"].startswith("I research topics")


# ---------- a report cut off by the model's output limit ----------
def test_truncated_report_is_trimmed_and_closed_with_limitations(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage

    from agent import nodes

    cut = ("# Market Report: EV Charging\n\n## Key Players\n\n* Tata Power leads [1].\n\n"
           "## Drivers, Barriers & Trends\n\n* **Government:** FAME India drives demand [1].\n"
           "* **Infrastructure:** stations are projected to expand at a CAGR of ~51.93% from FY 2")

    class Truncating:
        def invoke(self, prompt):
            return AIMessage(content=cut, response_metadata={"finish_reason": "length"})

    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: Truncating())
    monkeypatch.setattr(nodes, "REPORTS_DIR", tmp_path)
    f = {"question": "q", "answer": "a (source: https://a.com/x)", "sources": ["https://a.com/x"]}
    out = nodes.write({"goal": "g", "findings": [f], "critique": "Estimates vary.", "trace": []})
    report = out["report"]
    assert "from FY 2" not in report                      # half-sentence removed
    assert "FAME India drives demand" in report           # complete lines kept
    assert "## Limitations" in report and "output limit" in report and "Estimates vary." in report
    assert report.rstrip().endswith("1. https://a.com/x")  # sources still appended
    assert "output limit" in out["trace"][-1]


def test_feedback_is_kept_as_a_lesson_when_models_are_rate_limited(tmp_path, monkeypatch):
    from agent import llm, nodes

    class RateLimited:
        def with_structured_output(self, schema, **kw):
            return self

        def invoke(self, prompt):
            raise RuntimeError("429 rate limit")

    monkeypatch.setattr(llm, "PATIENCE_S", 0)
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: RateLimited())
    monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "m.sqlite")
    run_id = memory.save_run("g", "r")
    learned = nodes.learn_from_feedback(run_id, "g", 1, "  Always end with a recommendation  ")
    assert learned == ["Always end with a recommendation"]
    assert memory.get_lessons() == ["Always end with a recommendation"]


def test_qwen_keeps_reasoning_hidden_not_disabled():
    """Disabling qwen's reasoning broke its JSON output (planner, critic,
    feedback), so it must stay 'hidden'."""
    from agent.llm import _chat

    q = _chat("qwen/qwen3.8-27b", 0.0)
    assert q.reasoning_format == "hidden" and q.reasoning_effort is None and q.max_tokens <= 1000


def test_run_fails_fast_when_every_models_daily_quota_is_spent(tmp_path, monkeypatch):
    import time as _time

    from agent import llm, nodes

    class Spent:
        def invoke(self, prompt):
            raise RuntimeError("Error code: 429 - Rate limit reached ... tokens per day (TPD): Limit 200000")

    monkeypatch.setattr(llm, "_exhausted_until", {})
    monkeypatch.setattr(llm, "PATIENCE_S", 30)  # must NOT be waited on for a daily limit
    chain = llm._Chain([("a", Spent()), ("b", Spent())])

    class SpentLLM:
        def with_structured_output(self, schema, **kw):
            return chain

    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: SpentLLM())
    monkeypatch.setattr(memory, "MEMORY_DB", tmp_path / "m.sqlite")
    t0 = _time.time()
    with pytest.raises(llm.QuotaExhausted, match="used up for today"):
        run("Compare LangGraph and CrewAI for building agents")
    assert _time.time() - t0 < 5 and memory.count_runs() == 0
