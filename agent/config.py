"""Central configuration, read once from environment / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"  # runtime output; git-ignored along with data/
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Tried in order when the primary model is rate-limited or fails. Each Groq
# model has its own free-tier quota, so a chain keeps runs alive much longer.
GROQ_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "GROQ_FALLBACK_MODELS", os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b,qwen/qwen3.8-27b")
    ).split(",")
    if m.strip()
]
# gpt-oss models "think" before answering; low effort keeps runs fast.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")

MEMORY_DB = DATA_DIR / "memory.sqlite"

# Guard-rails that keep the agent bounded (and inside free-tier rate limits).
MAX_SUBQUESTIONS = int(os.getenv("MAX_SUBQUESTIONS", "3"))
MAX_TOOL_CALLS_PER_STEP = int(os.getenv("MAX_TOOL_CALLS_PER_STEP", "3"))
MAX_REFLECTION_ROUNDS = int(os.getenv("MAX_REFLECTION_ROUNDS", "1"))
PAGE_CHAR_LIMIT = int(os.getenv("PAGE_CHAR_LIMIT", "2500"))
