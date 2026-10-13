"""Recover saved first-case generations after a postprocessing-only schema bug.

No generation, search, retry, hyperparameter change or trajectory substitution.
"""
import copy
import json
from pathlib import Path

import run_v133_goal_al_wide_context as al
from mini_agent_rl.model.local import parse_agent_action


def main():
    out = al.OUT
    failure = al.read(out / "failure.json")
    count = al.read(out / "counts.json")
    if failure != dict(stage="evaluate-a", type="KeyError", error="'official_answer'", counts=count):
        raise RuntimeError("unexpected failure")
    if count != dict(generations=14, searches=0) or (out / "trajectory-A.jsonl").exists():
        raise RuntimeError("unexpected physical budget/result state")
    journal = [r for r in al.rows(out / "generation-journal.jsonl") if r["stage"] == "evaluate-A"]
    original = al.rows(al.assets.HIST)[0]
    if len(journal) != len(original["trace"]) or len(journal) != 4:
        raise RuntimeError("incomplete saved first-case outputs")
    result = copy.deepcopy(original)
    for step, saved in zip(result["trace"], journal):
        if step["prompt_token_ids"] != saved["prompt_token_ids"]:
            raise RuntimeError("first-case actual input mismatch; cannot recover")
        answer, tool = parse_agent_action(saved["raw"])
        if tool:
            if step["action"] != "search" or step["query"] != tool.arguments["query"]:
                raise RuntimeError("saved first-case search mismatch")
        elif step["action"] != "answer" or step["answer"] != answer:
            raise RuntimeError("saved first-case answer mismatch")
        for key in ("raw", "parse_error", "prompt_token_ids", "action_token_ids", "usage"):
            step[key] = saved[key]
        step["context_usage"] = saved["usage"]
        step["model_seconds"] = None
        if "observation" in step:
            step["observation"]["latency_ms"] = None
        step["recovery_basis"] = "saved AL actual input/action equality; tool text corroborated by later actual input IDs"
    result["elapsed_seconds"] = None
    result["recovered_from_saved_generations"] = True
    al.finalize_case(result, al.rows(al.assets.CASES)[0], "A")
    archive = out / "failed-answer-field-v1"
    archive.mkdir()
    old_manifest = al.read(out / "manifest.json")
    runner_path = str(Path(al.__file__))
    old_sha = old_manifest["hashes"][runner_path]
    for path, digest in old_manifest["hashes"].items():
        if path != runner_path and al.sha(Path(path)) != digest:
            raise RuntimeError("unrelated frozen asset changed")
    for name in ("failure.json", "status.json", "manifest.json", "failed-runner-source.py"):
        file = out / name
        if file.exists(): file.rename(archive / name)
    new_manifest = copy.deepcopy(old_manifest)
    new_manifest["hashes"][runner_path] = al.sha(Path(al.__file__))
    new_manifest["postprocessing_amendment"] = dict(old_runner_sha256=old_sha, new_runner_sha256=new_manifest["hashes"][runner_path],
        reason="source gold field is answer, not official_answer; no inference parameters changed")
    al.write(out / "manifest.json", new_manifest)
    al.append(out / "trajectory-A.jsonl", result)
    count["searches"] = result["searches"]
    al.write(out / "counts.json", count)
    al.write(out / "answer-field-recovery.json", dict(user_authorization="修复完就继续之前的任务", saved_generations=4,
        repeated_generations=0, next_case_index=1, verified_actual_inputs_and_actions=4, restored_physical_searches=result["searches"],
        unavailable_question_latencies=1, unavailable_generation_latencies=4,
        all_other_assets_unchanged=True, manifest_amendment=new_manifest["postprocessing_amendment"]))
    al.write(out / "status.json", dict(status="engineered"))
    print(json.dumps(al.read(out / "answer-field-recovery.json")))


if __name__ == "__main__":
    main()
