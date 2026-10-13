"""Hard-deadline wrapper for the explicitly authorized paired dev inference."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--authorize-model-inference", action="store_true")
    parser.add_argument("--max-minutes", type=int, default=60)
    parser.add_argument("--output", required=True)
    args, rest = parser.parse_known_args()
    if not args.authorize_model_inference:
        raise SystemExit("model inference requires separate explicit user authorization")
    if not 1 <= args.max_minutes <= 60:
        raise SystemExit("max-minutes must be in 1..60")
    output = Path(args.output)
    if output.exists():
        raise SystemExit("output_exists; automatic overwrite or retry is forbidden")
    command = [sys.executable, str(ROOT / "scripts/evaluate_v133_goal_c_sft.py"),
               *rest, "--output", str(output), "--max-minutes", str(args.max_minutes),
               "--authorize-model-inference"]
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
        subprocess.run(command, cwd=ROOT, env=env, check=True,
                       timeout=args.max_minutes * 60)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit("evaluation hard time limit reached; worker terminated, no retry") from exc
    if not output.is_file():
        raise SystemExit("evaluation returned without result file")
    state = json.loads(output.read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise SystemExit("evaluation incomplete; do not retry")


if __name__ == "__main__":
    main()
