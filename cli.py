"""Command-line entry point.

    python cli.py "Compare the top 3 open-source vector databases in 2026"
    python cli.py --mode market "EV charging market size in India"
    modes: general, competitor, market, leads
"""
import argparse
import sys

from agent.graph import run
from agent.modes import MODES
from agent.nodes import learn_from_feedback


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Autonomous research agent")
    parser.add_argument("--mode", choices=list(MODES), default="general")
    parser.add_argument("goal", nargs="*")
    args = parser.parse_args()
    goal = " ".join(args.goal).strip() or input("Research goal: ").strip()
    if not goal:
        sys.exit("Please provide a research goal.")

    def show(node, lines):
        for line in lines:
            print(f"\n[{node}] {line}", flush=True)

    try:
        state = run(goal, mode=args.mode, on_step=show)
    except ValueError as e:  # invalid input such as an over-long goal
        sys.exit(str(e))
    if not state.get("is_research", True):  # small talk: answered directly
        print("\n" + state["report"])
        return
    print("\n" + "=" * 70 + "\n" + state["report"] + "\n" + "=" * 70)
    print(f"Saved to {state['report_path']}")

    rating = input("\nWas this useful? (y/n, Enter to skip): ").strip().lower()
    if rating in ("y", "n"):
        feedback = input("Any feedback to improve future reports? ").strip()
        lessons = learn_from_feedback(state["run_id"], goal, 1 if rating == "y" else -1, feedback)
        for l in lessons:
            print(f"Learned: {l}")


if __name__ == "__main__":
    main()
