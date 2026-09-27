# AI Assistance Disclosure

The contest rules allow AI coding assistants for coding help. In the interest of transparency:

- **Assistant used:** an AI coding assistant, for scaffolding code, writing boilerplate
  (UI, API, tests) and drafting documentation.
- **My decisions:** choice of task (research agent); the workflow design (recall → plan →
  ReAct act/tools loop → record → reflect critic loop → write → remember); the tool set;
  using a free open-weights model on Groq; the "learn from feedback" memory mechanism; and the
  safety limits (tool budgets, reflection rounds).
- **Verification:** I ran the offline test suite, tested the agent live on several research
  goals, and reviewed the code so I can explain every component.
- **Runtime LLM:** Groq free tier (`openai/gpt-oss-120b`, open-weights). No paid APIs.
