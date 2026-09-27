"""Single place that builds the LLM client, so the provider is easy to swap.

Groq's free tier has token quotas *per model* (per minute and per day), so
every call goes through a fallback chain: if a model is rate-limited or
errors, the same request is retried on the next model. A model that reports
its *daily* quota is used up is skipped for a while instead of being retried
on every call.
"""
import time
from functools import lru_cache

from langchain_groq import ChatGroq

from .config import GROQ_API_KEY, GROQ_FALLBACK_MODELS, GROQ_MODEL, REASONING_EFFORT

DAILY_LIMIT_COOLDOWN_S = 15 * 60
PATIENCE_S = 60  # wait before one retry when every model failed (per-minute limits reset within a minute)
_exhausted_until: dict[str, float] = {}  # model name -> time it may be tried again


def _chat(model: str, temperature: float) -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    if "gpt-oss" in model:
        # gpt-oss models "think" before answering; low effort keeps runs fast and cheap.
        extra = {"reasoning_effort": REASONING_EFFORT}
    elif "qwen" in model:
        # its free tier allows only 1000 output tokens/minute: turn off hidden
        # "thinking" (which spends that budget) and ask for less than the cap
        extra = {"reasoning_effort": "none", "max_tokens": 950}
    else:
        extra = {}
    # max_retries handles short per-minute 429s with back-off.
    return ChatGroq(model=model, temperature=temperature, api_key=GROQ_API_KEY, max_retries=2, **extra)


def _is_daily_limit(error: Exception) -> bool:
    text = str(error)
    return "429" in text and ("tokens per day" in text or "requests per day" in text or "(TPD)" in text)


class _Chain:
    """Tries each model's runnable in order, skipping models whose daily quota is spent."""

    def __init__(self, named_runnables):
        self.named = named_runnables

    def invoke(self, prompt):
        last_error = None
        for name, runnable in self.named:
            if _exhausted_until.get(name, 0) > time.time():
                continue
            try:
                return runnable.invoke(prompt)
            except Exception as e:
                if _is_daily_limit(e):
                    _exhausted_until[name] = time.time() + DAILY_LIMIT_COOLDOWN_S
                last_error = e
        raise last_error or RuntimeError("All models are over their daily free-tier quota; try again later.")


class LLM:
    """Primary model plus fallbacks, exposing the three ways the nodes use an
    LLM: plain invoke, tool calling, and structured output."""

    def __init__(self, temperature: float):
        names = list(dict.fromkeys([GROQ_MODEL, *GROQ_FALLBACK_MODELS]))  # de-duplicate, keep order
        self.models = [(name, _chat(name, temperature)) for name in names]

    def invoke(self, prompt):
        return _Chain(self.models).invoke(prompt)

    def bind_tools(self, tools):
        return _Chain([(n, m.bind_tools(tools)) for n, m in self.models])

    def with_structured_output(self, schema, **kwargs):
        return _Chain([(n, m.with_structured_output(schema, **kwargs)) for n, m in self.models])


def invoke_with_patience(call):
    """Run `call()`; if every model failed (typically per-minute limits, which
    clear within a minute), wait once and try again before giving up."""
    try:
        return call()
    except Exception:
        time.sleep(PATIENCE_S)
        return call()


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.2) -> LLM:
    return LLM(temperature)
