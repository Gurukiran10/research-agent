"""Streamlit UI: shows the agent thinking step by step, then the report.

    streamlit run app.py
"""
import streamlit as st

from agent import memory
from agent.config import GROQ_MODEL
from agent.graph import run
from agent.nodes import learn_from_feedback

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")

NODE_LABELS = {
    "recall": "🧠 Recall memory",
    "plan": "🗺️ Plan",
    "act": "🤔 Reason / Act",
    "tools": "🛠️ Tool result",
    "record": "📌 Record finding",
    "reflect": "🔍 Reflect",
    "write": "✍️ Write report",
    "remember": "💾 Remember",
}

with st.sidebar:
    st.header("Long-term memory")
    st.caption(f"Model: `{GROQ_MODEL}` (Groq)")
    lessons = memory.get_lessons()
    st.subheader("Lessons learned from feedback")
    if lessons:
        for l in lessons:
            st.markdown(f"- {l}")
    else:
        st.caption("None yet - rate a report to teach the agent.")
    st.subheader("Past research")
    for r in memory.list_runs(10):
        icon = {1: "👍", -1: "👎"}.get(r["rating"], "•")
        st.markdown(f"{icon} {r['goal']}")

st.title("🔎 Autonomous Research Agent")
st.write("Give it a research goal. It **plans** sub-questions, **acts** with tools "
         "(web search, page reader, calculator, memory), **observes** results, "
         "**reflects** on gaps, and **writes** a cited report - and it learns from your feedback.")

goal = st.text_input("Research goal", placeholder="e.g. Compare LangGraph, CrewAI and AutoGen for building AI agents")

if st.button("Run agent", type="primary", disabled=not goal.strip()):
    st.session_state.pop("result", None)
    log = st.container(border=True)
    log.markdown("#### Agent workflow (live)")
    with st.spinner("Agent is working..."):
        def on_step(node, lines):
            for line in lines:
                log.markdown(f"**{NODE_LABELS.get(node, node)}** - {line}".replace("\n", "  \n"))
        try:
            st.session_state["result"] = run(goal.strip(), on_step=on_step)
            st.session_state["goal"] = goal.strip()
        except Exception as e:
            st.error(f"Agent failed: {e}")

result = st.session_state.get("result")
if result:
    tab_report, tab_plan, tab_trace = st.tabs(["📄 Report", "🗺️ Plan & findings", "🧾 Full trace"])
    with tab_report:
        st.markdown(result["report"])
        st.download_button("Download report (.md)", result["report"], file_name="report.md")
    with tab_plan:
        st.markdown(f"**Critic's verdict:** {result.get('critique', '')}")
        for i, f in enumerate(result["findings"], 1):
            with st.expander(f"Q{i}. {f['question']}"):
                st.markdown(f["answer"])
                if f["sources"]:
                    st.caption("Sources: " + " | ".join(f["sources"]))
    with tab_trace:
        st.code("\n".join(result["trace"]), language="text")

    st.divider()
    st.subheader("Teach the agent")
    col1, col2 = st.columns([1, 3])
    rating = col1.radio("Was this useful?", ["👍 Yes", "👎 No"], horizontal=True)
    feedback = col2.text_input("What should it do differently next time?",
                               placeholder="e.g. add a comparison table, prefer official docs as sources")
    if st.button("Submit feedback"):
        learned = learn_from_feedback(result["run_id"], st.session_state["goal"],
                                      1 if rating.startswith("👍") else -1, feedback)
        if learned:
            st.success("Learned: " + " / ".join(learned) + " - this will shape future reports.")
        else:
            st.info("Feedback saved.")
