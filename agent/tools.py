"""Tools the agent can choose to call while answering a sub-question.

Each tool is a plain Python function wrapped with @tool so the LLM sees its
name, docstring and arguments and can decide *when* to use it.
"""
import ast
import operator
import re
import time

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.tools import tool

from . import memory
from .config import PAGE_CHAR_LIMIT


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web (DuckDuckGo). Returns titles, URLs and short snippets.
    Use this first to discover relevant sources for a question."""
    # Free search engines rate-limit bursts; retry with backoff, then switch engines.
    results, error = [], None
    for attempt, backend in enumerate(["auto", "auto", "brave,mojeek,yahoo"]):
        try:
            results = DDGS().text(query, max_results=min(max_results, 5), backend=backend)
            if results:
                break
        except Exception as e:  # network / rate-limit issues should not crash the agent
            error = e
        time.sleep(1.5 * (attempt + 1))
    if not results:
        return f"Search failed ({error or 'no results'}). Try a shorter or rephrased query."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title', '')}\nURL: {r.get('href', '')}\n{r.get('body', '')}")
    return "\n\n".join(lines)


@tool
def read_webpage(url: str) -> str:
    """Download a web page and return its main readable text (truncated).
    Use this when a search snippet is not detailed enough."""
    try:
        resp = httpx.get(
            url,
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "research-agent/1.0 (educational project)", "Accept-Language": "en-US,en;q=0.9"},
        )
        resp.raise_for_status()
    except Exception as e:
        # Some sites block automated readers; report it so the agent picks another source.
        return f"Could not fetch page ({e.__class__.__name__}). Try a different URL or rely on search snippets."
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    return text[:PAGE_CHAR_LIMIT] or "Page had no readable text."


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression, e.g. '(120-80)/80*100'.
    Powers: use ** or ^ (CAGR = (end/start)**(1/years) - 1).
    Use this for growth rates, percentages, sums - never do math in your head."""
    try:
        # LLMs often write ^ for power; in Python ^ is XOR, so translate it.
        return str(round(_eval(ast.parse(expression.replace("^", "**"), mode="eval").body), 6))
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def search_past_research(query: str) -> str:
    """Look up reports this agent produced in earlier sessions.
    Use this to reuse knowledge you already gathered before searching the web."""
    hits = memory.find_related_reports(query, limit=3)
    if not hits:
        return "No related past research."
    return "\n\n".join(f"Past topic: {h['goal']}\n{h['report'][:1200]}" for h in hits)


TOOLS = [web_search, read_webpage, calculator, search_past_research]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
