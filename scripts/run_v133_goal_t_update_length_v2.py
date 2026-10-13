"""Goal T interface with CPU-only score recomputation in summarize."""
from __future__ import annotations

import argparse

import analyze_v133_goal_t_update_length as analysis
import run_v133_goal_t_update_length as goal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "summarize"])
    args = parser.parse_args()
    if args.command == "prepare": goal.prepare()
    elif args.command == "run":
        goal.run()
        if goal.read(goal.STATE)["status"] == "complete": analysis.main()
    else: analysis.main()


if __name__ == "__main__": main()
