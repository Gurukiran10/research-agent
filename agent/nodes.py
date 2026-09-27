"""Graph nodes. Each function takes the current state and returns an update.

Flow:  recall -> plan -> act <-> tools -> record -> (next step | reflect)
       reflect -> (more research | write) -> write -> remember
"""
import re
from datetime import date, datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from . import memory, prompts
from .config import (
    MAX_REFLECTION_ROUNDS, MAX_SUBQUESTIONS, MAX_TOOL_CALLS_PER_STEP, REPORTS_DIR,
)
from .llm import get_llm, invoke_with_patience
from .modes import get_mode
from .state import ResearchState
from .tools import TOOLS, TOOLS_BY_NAME

URL_RE = re.compile(r"https?://[^\s)\]\"'>,]+")


# ---------- structured-output schemas ----------
class Plan(BaseModel):
    sub_questions: list[str] = Field(description="Ordered, focused research sub-questions")


class Critique(BaseModel):
    sufficient: bool = Field(description="True if findings are enough for a good report")
    critique: str = Field(description="One or two sentences on quality / gaps")
    follow_up_questions: list[str] = Field(default_factory=list, description="At most 2 new sub-questions")


class Lessons(BaseModel):
    lessons: list[str] = Field(default_factory=list)


# ---------- helpers ----------
def _log(state: ResearchState, msg: str) -> list[str]:
    return state.get("trace", []) + [msg]


def _bullets(items: list[str], empty: str = "(none)") -> str:
    return "\n".join(f"- {i}" for i in items) if items else empty


def _structured(schema, prompt: str, temperature: float, default):
    """Ask for JSON matching `schema`. Uses Groq's JSON-schema mode, retries
    once, and falls back to a safe default so a malformed reply never crashes
    the workflow."""
    llm = get_llm(temperature).with_structured_output(schema, method="json_schema")
    try:
        return invoke_with_patience(lambda: llm.invoke(prompt))
    except Exception:
        return default


def _norm(url: str) -> str:
    return url.rstrip("/.;:").lower()


def _findings_text(state: ResearchState, max_chars: int = 900) -> str:
    out = []
    for i, f in enumerate(state.get("findings", []), 1):
        out.append(f"Q{i}: {f['question']}\nA: {f['answer'][:max_chars]}\nSources: {', '.join(f['sources']) or 'none'}")
    return "\n\n".join(out) or "(nothing yet)"


# ---------- nodes ----------
def recall(state: ResearchState) -> dict:
    """Load long-term memory relevant to this goal."""
    lessons = memory.get_lessons()
    related = memory.find_related_reports(state["goal"])
    summaries = [f"{r['goal']}: {r['report'][:300]}" for r in related]
    return {
        "lessons": lessons,
        "related_reports": summaries,
        "trace": _log(state, f"RECALL: mode={get_mode(state.get('mode')).label} | "
                             f"{len(lessons)} lesson(s), {len(summaries)} related past report(s)"),
    }


def plan(state: ResearchState) -> dict:
    """Break the goal into ordered sub-questions."""
    mode = get_mode(state.get("mode"))
    result = _structured(Plan, prompts.PLANNER.format(
        goal=state["goal"],
        today=date.today().isoformat(),
        mode_label=mode.label,
        mode_planning=mode.planning,
        max_q=MAX_SUBQUESTIONS,
        lessons=_bullets(state.get("lessons", [])),
        related=_bullets(state.get("related_reports", [])),
    ), 0.2, default=Plan(sub_questions=[state["goal"]]))
    questions = [q.strip() for q in result.sub_questions if q.strip()][:MAX_SUBQUESTIONS] or [state["goal"]]
    return {
        "plan": questions,
        "step_index": 0,
        "messages": [],
        "tool_calls_this_step": 0,
        "findings": [],
        "reflection_rounds": 0,
        "trace": _log(state, "PLAN:\n" + _bullets(questions)),
    }


def act(state: ResearchState) -> dict:
    """Reason about the current sub-question and either call a tool or answer."""
    idx = state["step_index"]
    question = state["plan"][idx]
    messages = list(state.get("messages", []))
    if not messages:  # starting a new sub-question -> fresh scratchpad with context
        previous = "\n".join(f"- {f['question']}: {f['answer'][:400]}" for f in state.get("findings", []))
        messages = [
            SystemMessage(prompts.EXECUTOR.format(
                today=date.today().isoformat(),
                mode_label=get_mode(state.get("mode")).label,
                mode_execution=get_mode(state.get("mode")).execution,
                goal=state["goal"], step=idx + 1, total=len(state["plan"]), question=question,
                previous=previous or "(this is the first sub-question)", budget=MAX_TOOL_CALLS_PER_STEP,
                lessons=_bullets(state.get("lessons", [])),
            )),
            HumanMessage(f"Research this sub-question: {question}"),
        ]

    budget_left = state.get("tool_calls_this_step", 0) < MAX_TOOL_CALLS_PER_STEP
    if budget_left:
        response = _invoke_with_tools(messages, state["goal"], question)
    else:
        response = _answer_without_tools(state["goal"], question, messages)

    if response.tool_calls:
        calls = ", ".join(f"{c['name']}({_short_args(c['args'])})" for c in response.tool_calls)
        note = f"ACT [Q{idx + 1}]: decided to call {calls}"
    else:
        note = f"ACT [Q{idx + 1}]: has enough evidence -> answering"
    return {"messages": messages + [response], "trace": _log(state, note)}


def _invoke_with_tools(messages, goal: str, question: str):
    """Tool-calling models occasionally emit malformed calls or invent tools
    that don't exist; retry once, then fall back to answering from the
    evidence so a single bad generation can't kill the run."""
    llm = get_llm(0.2).bind_tools(TOOLS)
    for _ in range(2):
        try:
            response = llm.invoke(messages)
        except Exception:  # groq returns 400 'tool_use_failed' on malformed calls
            continue
        if all(c["name"] in TOOLS_BY_NAME for c in response.tool_calls):
            return response
    return _answer_without_tools(goal, question, messages)


def _answer_without_tools(goal: str, question: str, messages) -> AIMessage:
    """Write the step's answer from the observations gathered so far.

    The evidence is passed as plain text (not as tool-call history) so the
    model has no reason to try calling a tool again."""
    evidence = "\n\n".join(
        f"[{m.name}] {str(m.content)[:2000]}" for m in messages if isinstance(m, ToolMessage)
    ) or "(no observations were gathered)"
    prompt = f"{prompts.FORCE_ANSWER}\n\nGoal: {goal}\nSub-question: {question}\n\nObservations:\n{evidence}"
    try:
        return AIMessage(content=get_llm(0.2).invoke(prompt).content)
    except Exception as e:
        return AIMessage(content=f"Could not synthesise an answer ({e.__class__.__name__}). Raw evidence:\n{evidence[:1500]}")


def _short_args(args: dict) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return text if len(text) < 90 else text[:87] + "..."


def run_tools(state: ResearchState) -> dict:
    """Execute every tool call requested by the last LLM message (Observe)."""
    last: AIMessage = state["messages"][-1]
    results, notes = [], []
    for call in last.tool_calls:
        tool = TOOLS_BY_NAME.get(call["name"])
        output = tool.invoke(call["args"]) if tool else f"Unknown tool {call['name']}"
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=call["name"]))
        notes.append(f"OBSERVE: {call['name']} returned {len(str(output))} chars")
    return {
        "messages": state["messages"] + results,
        "tool_calls_this_step": state.get("tool_calls_this_step", 0) + len(last.tool_calls),
        "trace": state.get("trace", []) + notes,
    }


def record(state: ResearchState) -> dict:
    """Save the answer for the current sub-question and reset the scratchpad."""
    idx = state["step_index"]
    answer = str(state["messages"][-1].content).strip()
    # Only trust URLs that a tool actually returned - the LLM can mistype or
    # invent URLs, so anything it "cites" that we never saw is dropped.
    seen = {}
    for m in state["messages"]:
        if isinstance(m, ToolMessage):
            for u in URL_RE.findall(str(m.content)):
                seen.setdefault(_norm(u), u)
    for m in state["messages"]:  # URLs the agent read directly are real too
        for c in getattr(m, "tool_calls", None) or []:
            if c["name"] == "read_webpage" and c["args"].get("url"):
                seen.setdefault(_norm(c["args"]["url"]), c["args"]["url"])
    cited = [seen[_norm(u)] for u in URL_RE.findall(answer) if _norm(u) in seen]
    if not cited:
        cited = list(seen.values())[:3]
    finding = {"question": state["plan"][idx], "answer": answer, "sources": list(dict.fromkeys(cited))}
    return {
        "findings": state.get("findings", []) + [finding],
        "step_index": idx + 1,
        "messages": [],
        "tool_calls_this_step": 0,
        "trace": _log(state, f"RECORD: finished Q{idx + 1} ({len(finding['sources'])} source(s))"),
    }


def reflect(state: ResearchState) -> dict:
    """Critique the findings; add follow-up questions if there are gaps."""
    result = _structured(
        Critique, prompts.REFLECTOR.format(
            goal=state["goal"], findings=_findings_text(state), today=date.today().isoformat(),
            mode_label=get_mode(state.get("mode")).label,
            mode_sections=", ".join(get_mode(state.get("mode")).sections),
        ), 0.0,
        default=Critique(sufficient=True, critique="(critic unavailable - proceeding to write)"),
    )
    rounds = state.get("reflection_rounds", 0)
    update = {"critique": result.critique, "reflection_rounds": rounds + 1}
    follow_ups = [q for q in result.follow_up_questions if q.strip()][:2]
    if not result.sufficient and follow_ups and rounds < MAX_REFLECTION_ROUNDS:
        update["plan"] = state["plan"] + follow_ups
        update["trace"] = _log(state, f"REFLECT: gaps found - {result.critique}\nAdding:\n" + _bullets(follow_ups))
    elif result.sufficient:
        update["trace"] = _log(state, f"REFLECT: findings sufficient - {result.critique}")
    else:
        update["trace"] = _log(state, f"REFLECT: reflection budget used - writing with noted limitations. {result.critique}")
    return update


def write(state: ResearchState) -> dict:
    """Compose the final cited report and save it to disk."""
    # Number every verified source once; the LLM cites by number only and the
    # code appends the Sources list, so every URL in the report is real.
    sources = list(dict.fromkeys(u for f in state.get("findings", []) for u in f["sources"]))
    num = {u: i for i, u in enumerate(sources, 1)}
    num_by_norm = {_norm(u): i for u, i in num.items()}

    def cite_inline(match):  # keep claim->source links: swap each verified URL for its [n]
        n = num_by_norm.get(_norm(match.group(0)))
        return f"[{n}]" if n else ""

    findings = "\n\n".join(
        f"Q{i}: {f['question']}\nA: {URL_RE.sub(cite_inline, f['answer'])[:1500]}\n"
        f"Sources for this answer: {' '.join(f'[{num[u]}]' for u in f['sources']) or '(none)'}"
        for i, f in enumerate(state.get("findings", []), 1)
    )
    mode = get_mode(state.get("mode"))
    note = "report saved"
    prompt = prompts.WRITER.format(
        goal=state["goal"],
        today=date.today().isoformat(),
        mode_label=mode.label,
        mode_report=mode.report,
        findings=findings,
        critique=state.get("critique", ""),
        lessons=_bullets(state.get("lessons", [])),
    )
    try:
        report = invoke_with_patience(lambda: get_llm(0.3).invoke(prompt)).content
    except Exception as e:
        # Every model is rate-limited: never lose a finished research run -
        # deliver the verified findings as the report instead.
        report = _report_from_findings(state, cite_inline, e)
        note = "writing model unavailable, report assembled from verified findings and saved"
    report = re.sub(r"【(\d+)[^】]*】", r"[\1]", report)  # normalise odd citation styles
    report = re.split(r"\n#+\s*Sources\b", report)[0].rstrip()  # drop any LLM-written source list
    report += "\n\n## Sources\n" + ("\n".join(f"{i}. {u}" for u, i in num.items()) or "_No sources._")
    slug = re.sub(r"[^a-z0-9]+", "-", state["goal"].lower()).strip("-")[:50] or "report"
    path = REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{mode.key}-{slug}.md"
    path.write_text(report, encoding="utf-8")
    return {"report": report, "report_path": str(path), "trace": _log(state, f"WRITE: {note} to {path.name}")}


def _report_from_findings(state: ResearchState, cite_inline, error: Exception) -> str:
    parts = [
        f"# Research notes: {state['goal']}",
        f"> The writing model was unavailable ({type(error).__name__}, usually a free-tier rate limit), "
        "so these are the agent's verified findings, unedited. Run again later for a polished report.",
    ]
    for i, f in enumerate(state.get("findings", []), 1):
        parts.append(f"## Q{i}. {f['question']}\n\n{URL_RE.sub(cite_inline, f['answer'])}")
    if state.get("critique"):
        parts.append(f"## Limitations\n\n{state['critique']}")
    return "\n\n".join(parts)


def remember(state: ResearchState) -> dict:
    """Persist the run so future research can reuse it."""
    details = {k: state.get(k) for k in ("plan", "findings", "critique", "reflection_rounds", "report_path")}
    details["trace"] = _log(state, "REMEMBER: stored in long-term memory")
    run_id = memory.save_run(state["goal"], state["report"], state.get("mode", "general"), details)
    return {"run_id": run_id, "trace": _log(state, f"REMEMBER: stored as run #{run_id}")}


# ---------- routers (conditional edges) ----------
def after_act(state: ResearchState) -> str:
    return "tools" if state["messages"][-1].tool_calls else "record"


def after_record(state: ResearchState) -> str:
    return "act" if state["step_index"] < len(state["plan"]) else "reflect"


def after_reflect(state: ResearchState) -> str:
    return "act" if state["step_index"] < len(state["plan"]) else "write"


# ---------- learning from feedback (called by the UI after a run) ----------
def learn_from_feedback(run_id: int, goal: str, rating: int, feedback: str) -> list[str]:
    memory.rate_run(run_id, rating, feedback)
    if not feedback.strip():
        return []
    result = _structured(Lessons, prompts.LESSON_EXTRACTOR.format(
        rating_word="positively" if rating > 0 else "negatively", feedback=feedback, goal=goal,
    ), 0.0, default=Lessons())
    for lesson in result.lessons[:2]:
        memory.add_lesson(lesson)
    return result.lessons[:2]
