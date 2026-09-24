"""Launch independent processes to audit every retained v094/v095 adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/qwen_08b")
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapters", nargs="+", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"输出已存在：{output}")
    verifier = Path(__file__).with_name("run_grpo_acceptance_v094.py").resolve()
    runs = {}
    for adapter_text in args.adapters:
        adapter = Path(adapter_text)
        name = adapter.name
        attempts = []
        for attempt in range(2):
            attempt_output = output.parent / f"{output.stem}-{name}-load{attempt}.json"
            completed = subprocess.run([sys.executable, str(verifier), "--verify-adapter", "--model", args.model,
                                        "--adapter", str(adapter), "--source-db", args.source_db, "--output", str(attempt_output)],
                                       text=True, capture_output=True, timeout=600, check=False)
            if completed.returncode != 0 or not attempt_output.exists():
                attempts.append({"status": "FAILED", "stderr": completed.stderr[-1000:]})
                continue
            attempts.append(json.loads(attempt_output.read_text(encoding="utf-8")))
        passed = len(attempts) == 2 and all(item.get("status") == "PASSED" for item in attempts)
        if passed:
            passed = (attempts[0]["adapter_file_digests"] == attempts[1]["adapter_file_digests"]
                      and attempts[0]["scores"] == attempts[1]["scores"])
        runs[name] = {"adapter_path": str(adapter), "adapter_file_sha256": sha(adapter / "adapter_model.safetensors"),
                      "attempts": attempts, "passed": passed}
    report = {"version": "v0.9.5", "source_db_sha256": sha(Path(args.source_db)), "runs": runs,
              "all_passed": all(item["passed"] for item in runs.values())}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_passed": report["all_passed"], "output": str(output)}))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
