"""Tools the agent can choose to call while answering a sub-question.

Each tool is a plain Python function wrapped with @tool so the LLM sees its
name, docstring and arguments and can decide *when* to use it.
"""
import ast
import ipaddress
import operator
import re
import socket
import time
from urllib.parse import urlparse

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


_NAT64 = ipaddress.ip_network("64:ff9b::/96")
MAX_PAGE_BYTES = 3_000_000
MAX_REDIRECTS = 5


def _public_http_url(url: str) -> str | None:
    """Return a reason the URL must not be fetched, or None if it is a public
    http(s) address. Blocks localhost / private networks so web content can't
    steer the agent into reading internal services (SSRF)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return "only public http(s) URLs can be read"
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except OSError:
        return "the host name could not be resolved"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        # NAT64 (64:ff9b::/96, common on mobile networks) and IPv4-mapped IPv6
        # addresses wrap an IPv4 address - judge the address inside
        if ip.version == 6 and ip in _NAT64:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        elif ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return "local or private network addresses are not allowed"
    return None


def _fetch(url: str) -> tuple[httpx.Response | None, str | None]:
    headers = {"User-Agent": "research-agent/1.0 (educational project)", "Accept-Language": "en-US,en;q=0.9"}
    with httpx.Client(timeout=15, follow_redirects=False, headers=headers) as client:
        for _ in range(MAX_REDIRECTS + 1):
            if reason := _public_http_url(url):
                return None, reason
            resp = client.get(url)
            if resp.is_redirect and resp.headers.get("location"):
                url = str(resp.next_request.url) if resp.next_request else resp.headers["location"]
                continue  # every redirect target is validated again
            return resp, None
    return None, "too many redirects"


@tool
def read_webpage(url: str) -> str:
    """Download a web page and return its main readable text (truncated).
    Use this when a search snippet is not detailed enough. HTML pages only."""
    fail = "Could not read page ({}). Try a different URL or rely on search snippets."
    try:
        resp, reason = _fetch(url.strip())
    except Exception as e:  # network errors should not crash the agent
        return fail.format(f"{type(e).__name__}: {str(e)[:80]}")
    if reason:
        return fail.format(reason)
    if resp.status_code >= 400:
        # Some sites block automated readers; report it so the agent picks another source.
        return fail.format(f"HTTP {resp.status_code}")
    content_type = resp.headers.get("content-type", "").lower()
    if "html" not in content_type and "text/plain" not in content_type:
        return fail.format(f"unsupported content type {content_type.split(';')[0] or 'unknown'}")
    if len(resp.content) > MAX_PAGE_BYTES:
        return fail.format("page too large")
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


MAX_EXPONENT = 1000  # 9**9**9 would otherwise hang the process computing a huge integer


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent too large (max {MAX_EXPONENT})")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression, e.g. '(120-80)/80*100'.
    Powers: use ** or ^ (CAGR = (end/start)**(1/years) - 1).
    Use this for growth rates, percentages, sums - never do math in your head."""
    expr = expression.strip()
    expr = expr.replace("^", "**")  # LLMs often write ^ for power; in Python ^ is XOR
    expr = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expr)  # thousands separators: 1,240 -> 1240
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d(])", r"(\1/100)", expr)  # 46.3% -> (46.3/100); 10 % 3 stays modulo
    if not expr:
        return "Calculation error: empty expression"
    try:
        return str(round(_eval(ast.parse(expr, mode="eval").body), 6))
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
