"""Research modes: the same agent engine, specialised for different business tasks.

A mode changes *how* the agent plans, which evidence it hunts for, and the shape
of the final report - without changing the workflow graph itself.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    description: str
    example: str
    planning: str     # what the planner must cover
    execution: str    # how the executor should gather evidence
    report: str       # Markdown structure for the writer

    @property
    def sections(self) -> list[str]:
        """Section names of the report, e.g. ['Competitor Comparison', ...]."""
        return [line[3:].split(" - ")[0].strip() for line in self.report.splitlines() if line.startswith("## ")]


MODES: dict[str, Mode] = {m.key: m for m in [
    Mode(
        key="general",
        label="General Research",
        description="Any topic: search, read, and compile a cited report.",
        example="Compare LangGraph, CrewAI and AutoGen for building AI agents",
        planning="Cover background first, then specifics, then comparisons or outlook.",
        execution="Prefer authoritative sources (official docs, reputable publications).",
        report="""# <clear title>
**TL;DR** - 2-3 sentence answer to the goal.
## Key Findings  - 4-7 bullets, each ending with a citation like [1]
## Details       - short sections (a compact table is fine if useful)
## Limitations   - what could not be verified, including the reviewer's gaps""",
    ),
    Mode(
        key="competitor",
        label="Competitor Intelligence",
        description="Profile a company and its competitors: products, pricing, positioning, hiring, recent launches.",
        example="Who are the main competitors of Zoho CRM and how do they compare on pricing and features?",
        planning=(
            "Identify the target company and its 3-4 closest competitors. Plan sub-questions on: "
            "(a) products and positioning, (b) pricing and plans, (c) recent launches, funding or news "
            "in the last 12 months, (d) hiring signals / growth areas."
        ),
        execution=(
            "Prefer company websites, pricing pages, press releases, and recent news. Note the date of "
            "any news item. Hiring signals can come from careers pages or job-post news."
        ),
        report="""# Competitive Landscape: <company / market>
**TL;DR** - who leads, who is cheapest, who is moving fastest (2-3 sentences).
## Competitor Comparison  - a Markdown table: Company | Positioning | Pricing | Recent moves [n]
## Company Profiles       - one short subsection per company with citations
## Strategic Signals      - launches, funding, hiring trends worth watching [n]
## Opportunities & Threats - 3-5 bullets for the target company
## Limitations""",
    ),
    Mode(
        key="market",
        label="Market Research",
        description="Size a market (TAM/SAM/SOM), growth rate, key players, drivers and trends.",
        example="What is the size and growth rate of the EV charging market in India?",
        planning=(
            "Plan sub-questions on: (a) current market size from 2+ sources, (b) forecast and CAGR, "
            "(c) key players and market shares, (d) growth drivers, barriers and trends."
        ),
        execution=(
            "Collect numeric estimates WITH their year, currency and source. Always use the calculator "
            "to compute or check CAGR = (end/start)**(1/years) - 1 and any TAM/SAM/SOM arithmetic. "
            "When sources disagree, record each figure separately."
        ),
        report="""# Market Report: <market>
**TL;DR** - market size, growth rate, and the one-line outlook.
## Market Size & Growth  - table: Source | Year | Size | Forecast | CAGR [n]; state calculations explicitly
## TAM / SAM / SOM       - estimate only if the findings support it, showing the arithmetic
## Key Players           - bullets or table [n]
## Drivers, Barriers & Trends - bullets [n]
## Limitations           - note conflicting estimates and assumptions""",
    ),
    Mode(
        key="leads",
        label="Lead Research",
        description="Build a prospect list from an ideal-customer profile, with fit reasons and buying signals.",
        example="Engineering colleges in Bangalore that offer AI/ML programs and run placement drives",
        planning=(
            "Turn the ideal-customer profile into sub-questions that (a) discover candidate organisations, "
            "(b) collect firmographics (location, size, type), (c) find buying signals or needs (recent "
            "initiatives, programs, hiring, expansions), (d) find public contact routes (official website, "
            "department or general contact pages)."
        ),
        execution=(
            "Find real, named organisations and their official websites. Only record public, "
            "organisation-level contact routes (website, contact page) - never guess personal emails "
            "or phone numbers."
        ),
        report="""# Lead List: <ideal-customer profile>
**TL;DR** - how many leads were found and the top 3 to contact first.
## Lead Table   - Markdown table: # | Organisation | Location | Fit reason | Buying signal | Source; the Source cell is the [n] backing that row. Leave a cell as "not found" rather than guessing
## Scoring      - rank leads High / Medium / Low fit with a one-line reason each
## Suggested Outreach Angle - 2-3 bullets on what to pitch based on the signals
## Limitations  - what could not be verified""",
    ),
]}

DEFAULT_MODE = "general"


def get_mode(key: str | None) -> Mode:
    return MODES.get(key or DEFAULT_MODE, MODES[DEFAULT_MODE])
