# Demo video script (target: 3-4 minutes)

Record with **Windows + Alt + R** (Xbox Game Bar) or OBS. Speak slowly. Upload to
YouTube as **Unlisted** (or Google Drive, "Anyone with the link can view").

Tip: a full run takes ~3-4 min. Either let it run and speed that part up 2-4x in the
editor, or start the run, keep talking over the live log, and cut the waiting.

---

**0:00-0:20 | Intro** (show the README on GitHub)
> "Hi, I'm Gurukiran. For the AI Agentic System Challenge I built an autonomous
> Research Agent. You give it a research goal, and it plans, uses tools, checks its own
> work, and writes a cited report. It also learns from feedback, so it gets better with use."

**0:20-0:50 | Architecture** (scroll to the mermaid diagram)
> "It's a LangGraph state machine. First it recalls long-term memory, then a planner
> breaks the goal into sub-questions. For each one, the agent runs a ReAct loop: it
> reasons, calls a tool like web search, the page reader, or the calculator, observes the
> result, and decides again. After all questions, a critic node checks for gaps and can
> add follow-up questions. That's a conditional loop, not a fixed pipeline. Finally
> it writes the report and stores it in memory."

**0:50-2:30 | Live run** (Streamlit UI: `streamlit run app.py`)
- Type a goal, e.g. *"How big is the global AI agents market in 2025 and what CAGR is forecast to 2030?"*
- Point at the live log:
  > "Here it recalled one related past report from memory... here's the plan it made...
  > now it decided on its own to call web search... it read this page... here it used the
  > calculator instead of doing math in its head..."
- When REFLECT appears:
  > "The critic found a gap, so the agent added new questions and went back to research.
  > This is the self-correction step."

**2:30-3:10 | Output**
- Report tab: TL;DR, key findings with [n] citations, limitations, sources.
  > "Every source URL is verified in code. Only URLs the tools actually returned are
  > allowed, so the LLM can't invent citations."
- Plan & findings tab: each sub-question with its sources.
- Full trace tab: every decision the agent made.

**3:10-3:40 | Learning from feedback**
- Choose 👎/👍, type feedback such as *"Always include a comparison table and prefer official sources"*, click **Submit feedback**.
- Show the "Learned: ..." message and the sidebar **Lessons learned** list.
  > "This lesson is saved in SQLite and injected into the planner and writer next time,
  > so the agent adapts to the user without any retraining."

**3:40-4:00 | Wrap-up**
> "Built with Python, LangGraph, and the free Groq API using the open-weights gpt-oss
> model. There's also a CLI, a FastAPI endpoint, and offline tests. Thank you!"
