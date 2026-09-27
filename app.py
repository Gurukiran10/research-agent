"""Streamlit UI: shows the agent working step by step, then the report.

    streamlit run app.py
"""
import html
import re

import streamlit as st

from agent import memory
from agent.config import GROQ_MODEL
from agent.graph import run
from agent.modes import MODES, get_mode
from agent.nodes import learn_from_feedback

st.set_page_config(page_title="Research Agent", page_icon=":material/travel_explore:", layout="centered")

# (tag text, tag colour) for each graph node in the live timeline
STEP_TAGS = {
    "recall": ("MEMORY", "#6B7280"),
    "plan": ("PLAN", "#13203A"),
    "act": ("THINK", "#2F5DA8"),
    "tools": ("TOOL", "#B45309"),
    "record": ("FOUND", "#2F7D5B"),
    "reflect": ("CRITIC", "#7A3E9D"),
    "write": ("WRITE", "#13203A"),
    "remember": ("SAVED", "#6B7280"),
}
TRACE_PREFIX = {  # trace lines start with these words; map them back to nodes
    "RECALL": "recall", "PLAN": "plan", "ACT": "act", "OBSERVE": "tools",
    "RECORD": "record", "REFLECT": "reflect", "WRITE": "write", "REMEMBER": "remember",
}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], button, input, textarea, p, li {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
footer, [data-testid="stDecoration"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 920px; padding-top: 3.5rem; padding-bottom: 4rem; }
h1 { font-weight: 600 !important; letter-spacing: -0.02em; }
/* colours inherit the active theme text colour so light and dark modes both work */
.eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: .14em;
  text-transform: uppercase; color: #C26A2A; margin-bottom: .25rem; }
.lede { opacity: .78; font-size: 1.05rem; line-height: 1.55; margin-bottom: 1.5rem; }
.mode-desc { font-size: 14.5px; opacity: .78; margin: -.25rem 0 .15rem; }
.sections { font-family: 'JetBrains Mono', monospace; font-size: 12px; opacity: .6; margin: 0 0 1rem; }
.model { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; }
.step { display: flex; gap: 14px; align-items: flex-start; padding: 7px 0;
  border-bottom: 1px solid rgba(128,128,128,.18); font-size: 14px; line-height: 1.5; }
.step:last-child { border-bottom: none; }
.tag { font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 500; color: #fff;
  padding: 2px 0; border-radius: 4px; min-width: 64px; text-align: center; margin-top: 2px; }
.step code { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; background: rgba(128,128,128,.14);
  color: inherit; padding: 1px 5px; border-radius: 3px; }
.side-h { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: .12em;
  text-transform: uppercase; opacity: .6; margin: 1.25rem 0 .5rem; }
.lesson { font-size: 13.5px; line-height: 1.45; padding: 8px 10px; background: rgba(128,128,128,.08);
  border: 1px solid rgba(128,128,128,.22); border-radius: 6px; margin-bottom: 6px; }
.hist { font-size: 13.5px; line-height: 1.4; padding: 7px 0; border-bottom: 1px solid rgba(128,128,128,.2); }
.hist .meta { display: block; font-family: 'JetBrains Mono', monospace; font-size: 11px; opacity: .6; margin-top: 2px; }
.muted { opacity: .65; font-size: 13.5px; }
.wait-note { font-size: 13.5px; opacity: .7; margin: .25rem 0 .5rem; }
[data-testid="stHeaderActionElements"] { display: none; }
[data-testid="stTabPanel"] h1 { font-size: 1.75rem !important; line-height: 1.25; }
[data-testid="stTabPanel"] h2 { font-size: 1.3rem !important; margin-top: 1.25rem; }
[data-testid="stTabPanel"] table { font-size: 14px; }
</style>
""",
    unsafe_allow_html=True,
)


def step_html(node: str, line: str) -> str:
    """One row of the timeline. Trace text comes from web pages and the LLM,
    so it is always HTML-escaped before rendering."""
    tag, colour = STEP_TAGS.get(node, (node.upper(), "#6B7280"))
    text = line.split(": ", 1)[1] if re.match(r"^[A-Z]+( \[Q\d+\])?: ", line) else line
    q = re.match(r"^[A-Z]+ \[(Q\d+)\]", line)
    body = html.escape(text).replace("\n", "<br>")
    body = re.sub(r"(\w+)\((.*?)\)", r"<code>\1(\2)</code>", body, count=1) if node == "act" else body
    prefix = f"<b>{q.group(1)}</b> · " if q else ""
    return f'<div class="step"><span class="tag" style="background:{colour}">{tag}</span><div>{prefix}{body}</div></div>'


def node_of(line: str) -> str:
    return TRACE_PREFIX.get(re.split(r"[ :\[]", line, maxsplit=1)[0], "act")


# ---------------- sidebar: long-term memory ----------------
with st.sidebar:
    st.markdown('<p class="side-h">Model</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="muted"><span class="model">{html.escape(GROQ_MODEL)}</span> on Groq</p>', unsafe_allow_html=True)

    st.markdown('<p class="side-h">Learned preferences</p>', unsafe_allow_html=True)
    lessons = memory.get_lessons()
    if lessons:
        st.markdown("".join(f'<div class="lesson">{html.escape(l)}</div>' for l in lessons), unsafe_allow_html=True)
    else:
        st.markdown('<p class="muted">Nothing yet. Rate a report to teach the agent.</p>', unsafe_allow_html=True)

    st.markdown('<p class="side-h">Research history</p>', unsafe_allow_html=True)
    runs = memory.list_runs(8)
    if not runs:
        st.markdown('<p class="muted">No research yet.</p>', unsafe_allow_html=True)
    for r in runs:
        rating = {1: " · rated helpful", -1: " · rated not helpful"}.get(r["rating"], "")
        st.markdown(
            f'<div class="hist">{html.escape(r["goal"])}'
            f'<span class="meta">{html.escape(get_mode(r.get("mode")).label)}{rating}</span></div>',
            unsafe_allow_html=True,
        )

# ---------------- main: input ----------------
st.markdown('<p class="eyebrow">Autonomous research agent</p>', unsafe_allow_html=True)
st.title("What should I research?")
st.markdown(
    '<p class="lede">The agent plans sub-questions, gathers evidence with web search, a page reader and a '
    "calculator, has a critic check the findings, and writes a cited report. Your feedback becomes "
    "preferences it follows next time.</p>",
    unsafe_allow_html=True,
)

mode_key = st.segmented_control(
    "Research mode", list(MODES), format_func=lambda k: MODES[k].label,
    default="general", key="mode", label_visibility="collapsed",
) or "general"
mode = MODES[mode_key]
st.markdown(
    f'<p class="mode-desc">{html.escape(mode.description)}</p>'
    f'<p class="sections">Report: {html.escape(" · ".join(mode.sections))}</p>',
    unsafe_allow_html=True,
)

if st.button("Use an example goal", type="tertiary", icon=":material/lightbulb:"):
    st.session_state["goal_input"] = mode.example
goal = st.text_area(
    "Research goal", placeholder=mode.example, key="goal_input", height=90, label_visibility="collapsed",
)

if st.button("Run research", type="primary", icon=":material/play_arrow:", disabled=not goal.strip()):
    st.session_state.pop("result", None)
    st.markdown('<p class="wait-note">A full run usually takes 3–5 minutes (longer if the critic sends the '
                "agent back for more research). The report appears below when it finishes.</p>",
                unsafe_allow_html=True)
    with st.status("Recalling memory…", expanded=True) as status:
        progress = {"total": 0}

        def on_step(node, lines):
            for line in lines:
                st.markdown(step_html(node, line), unsafe_allow_html=True)
                # plan / reflect list new sub-questions as "- ..." lines
                progress["total"] += sum(1 for l in line.splitlines() if l.startswith("- "))
                q = re.match(r"^ACT \[Q(\d+)\]", line)
                if node == "plan":
                    status.update(label=f"Planned {progress['total']} sub-questions")
                elif q:
                    status.update(label=f"Researching question {q.group(1)} of {progress['total']}…")
                elif node == "reflect":
                    status.update(label="Critic is reviewing the findings…")
                elif node == "write":
                    status.update(label="Report written, saving…")
            if node == "reflect" and not any("gaps found" in l for l in lines):
                status.update(label="Writing the report…")
        try:
            result = run(goal.strip(), mode=mode_key, on_step=on_step)
            st.session_state["result"] = result
            st.session_state["goal"] = goal.strip()
            status.update(label=f"Done · {len(result['trace'])} steps", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Research failed", state="error", expanded=True)
            st.error(f"{type(e).__name__}: {e}")

# ---------------- main: results ----------------
result = st.session_state.get("result")
if result:
    trace = result["trace"]
    sources = {u for f in result["findings"] for u in f["sources"]}
    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sub-questions", len(result["plan"]))
    c2.metric("Tool calls", sum(1 for t in trace if t.startswith("OBSERVE")))
    c3.metric("Verified sources", len(sources))
    c4.metric("Critic rounds", result.get("reflection_rounds", 0))

    tab_report, tab_findings, tab_trace = st.tabs(["Report", "Findings & sources", "Agent trace"])
    with tab_report:
        with st.container(border=True):
            st.markdown(result["report"])
        st.download_button("Download report (.md)", result["report"], file_name="report.md",
                           icon=":material/download:")
    with tab_findings:
        st.markdown(f"**Critic's verdict:** {result.get('critique', '')}")
        for i, f in enumerate(result["findings"], 1):
            with st.expander(f"Q{i} · {f['question']}"):
                st.markdown(f["answer"])
                if f["sources"]:
                    st.caption("Sources: " + " · ".join(f["sources"]))
    with tab_trace:
        st.markdown("".join(step_html(node_of(t), t) for t in trace), unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="eyebrow">Teach the agent</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    helpful = col1.segmented_control("Was this useful?", ["Helpful", "Not helpful"], default="Helpful")
    feedback = col2.text_input("What should it do differently next time?",
                               placeholder="e.g. add a comparison table, prefer official sources")
    if st.button("Save feedback", icon=":material/school:"):
        learned = learn_from_feedback(result["run_id"], st.session_state["goal"],
                                      -1 if helpful == "Not helpful" else 1, feedback)
        if learned:
            st.success("Learned: " + " / ".join(learned) + ". This will shape future reports.")
        else:
            st.info("Feedback saved.")
