"""Single place that builds the LLM client, so the provider is easy to swap.

Groq's free tier has a daily token quota *per model*, so every call is wrapped
with an automatic fallback to a second model: if the primary is rate-limited
or errors, the same request is retried on the fallback.
"""
from functools import lru_cache

from langchain_groq import ChatGroq

from .config import GROQ_API_KEY, GROQ_FALLBACK_MODEL, GROQ_MODEL, REASONING_EFFORT


def _chat(model: str, temperature: float) -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    # gpt-oss models "think" before answering; low effort keeps runs fast and cheap.
    extra = {"reasoning_effort": REASONING_EFFORT} if "gpt-oss" in model else {}
    # max_retries handles short per-minute 429s with back-off.
    return ChatGroq(model=model, temperature=temperature, api_key=GROQ_API_KEY, max_retries=2, **extra)


class LLM:
    """Primary model with automatic fallback, exposing the three ways the
    nodes use an LLM: plain invoke, tool calling, and structured output."""

    def __init__(self, temperature: float):
        self.models = [_chat(GROQ_MODEL, temperature)]
        if GROQ_FALLBACK_MODEL and GROQ_FALLBACK_MODEL != GROQ_MODEL:
            self.models.append(_chat(GROQ_FALLBACK_MODEL, temperature))

    @staticmethod
    def _chain(runnables):
        return runnables[0].with_fallbacks(runnables[1:]) if len(runnables) > 1 else runnables[0]

    def invoke(self, prompt):
        return self._chain(self.models).invoke(prompt)

    def bind_tools(self, tools):
        return self._chain([m.bind_tools(tools) for m in self.models])

    def with_structured_output(self, schema, **kwargs):
        return self._chain([m.with_structured_output(schema, **kwargs) for m in self.models])


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.2) -> LLM:
    return LLM(temperature)
