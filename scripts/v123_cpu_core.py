"""Read-only inputs and fail-closed checks for the CPU-only v123 work.

This module never imports a generator, trainer, or an API client.  It only
opens explicitly named, unsealed inputs; generated artifacts live in v123.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from zipfile import ZipFile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/evidence-sft-v123"
RISKS = ("relation", "time", "scope", "measurement")
FORBIDDEN = ("final-test", "final_test", "independent", "holdout")


def safe(path: Path, *, expect: Path = ROOT) -> Path:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(expect.resolve()):
        raise ValueError("path_outside_allowed_root")
    if any(any(token in part.casefold() for token in FORBIDDEN) for part in resolved.parts):
        raise ValueError("sealed_path_forbidden")
    return resolved


def validate_archive_members(path: Path, *, expect: Path = ROOT) -> list[str]:
    """Reject sealed members, traversal, aliases and duplicate names pre-extraction."""
    result = []
    seen = set()
    with ZipFile(safe(path, expect=expect)) as archive:
        for member in archive.infolist():
            parts = Path(member.filename.replace("\\", "/")).parts
            if (member.filename.startswith(("/", "\\")) or ":" in member.filename or
                    any(part in {"", ".", ".."} for part in parts) or
                    any(any(token in part.casefold() for token in FORBIDDEN) for part in parts)):
                raise ValueError("forbidden_archive_member")
            key = member.filename.replace("\\", "/").casefold()
            if key in seen:
                raise ValueError("duplicate_archive_member")
            seen.add(key)
            result.append(member.filename)
    return result


def sha(path: Path) -> str:
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(safe(path).read_text(encoding="utf-8"))


def readl(path: Path) -> list[dict]:
    return [json.loads(line) for line in safe(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, value) -> None:
    safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def writel(path: Path, rows) -> None:
    safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def norm(value: str) -> str:
    import unicodedata
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in value).split())


def question(action_row: dict) -> str:
    return next((m["content"] for m in action_row["messages"] if m["role"] == "user"), "")


def action(action_row: dict) -> str:
    return json.loads(action_row["messages"][action_row["target_assistant_index"]]["content"])["action"]


def nonempty_search(action_row: dict) -> bool:
    if action(action_row) != "search":
        return False
    for message in action_row["messages"][:action_row["target_assistant_index"]]:
        if message["role"] != "tool":
            continue
        try:
            data = json.loads(message["content"])
        except (TypeError, json.JSONDecodeError):
            continue
        if any(isinstance(doc, dict) and doc.get("text") for doc in data.get("result") or []):
            return True
    return False


def export_packet(packet: dict) -> dict:
    messages = packet["visible_messages"]
    target = {"role": "assistant", "content": json.dumps(packet["proposed_action"],
              ensure_ascii=False, separators=(",", ":"))}
    return {"messages": messages + [target], "target_assistant_index": len(messages),
            "thinking_enabled": False}


def select_risk_pairs(pairs: list[tuple[str, list]], count: int = 50) -> list | None:
    """Select the first feasible set in the frozen candidate order."""
    risks = [Counter(tag for _, labels in records for tag in labels) for _, records in pairs]
    suffix = [Counter() for _ in range(len(pairs) + 1)]
    for i in range(len(pairs) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + risks[i]
    selected = []
    failed = set()

    def visit(i: int, need: tuple[int, ...]) -> bool:
        if len(selected) == count:
            return all(n == 0 for n in need)
        if len(pairs) - i < count - len(selected) or any(suffix[i][r] < n for r, n in zip(RISKS, need)):
            return False
        key = (i, count - len(selected), need)
        if key in failed:
            return False
        selected.append(pairs[i])
        newer = tuple(max(0, n - risks[i][r]) for r, n in zip(RISKS, need))
        if visit(i + 1, newer):
            return True
        selected.pop()
        if visit(i + 1, need):
            return True
        failed.add(key)
        return False

    return list(selected) if visit(0, (5, 5, 5, 5)) else None


def validate_selected_pairs(packets: list[dict], source: dict[str, str], selected: list[str],
                            risks: dict[str, list[str]], train_questions: set[str],
                            history_ids: set[str], history_text: set[str]) -> dict:
    by_question: dict[str, list[dict]] = defaultdict(list)
    for packet in packets:
        if packet["case_id"] not in source:
            raise ValueError("missing_case_source")
        by_question[source[packet["case_id"]]].append(packet)
    names = [norm(by_question[q][0]["question"]) for q in selected if q in by_question]
    counts = Counter(label for q in selected for packet in by_question.get(q, [])
                     for label in risks.get(packet["case_id"], []))
    states = [packet for q in selected for packet in by_question.get(q, [])]
    paired = all(len(by_question.get(q, [])) == 2 and
                 {p["proposed_action"]["action"] for p in by_question[q]} == {"search", "answer"}
                 for q in selected)
    checks = {
        "fifty_distinct_questions": len(selected) == len(set(selected)) == 50,
        "hundred_states_two_actions_per_question": paired and len(states) == 100,
        "four_semantic_risks_at_least_five": all(counts[r] >= 5 for r in RISKS),
        "question_text_unique": len(names) == len(set(names)) == 50,
        "historical_id_isolation": not set(selected) & history_ids,
        "historical_text_isolation": not set(names) & history_text,
        "training_text_isolation": not set(names) & train_questions,
        "unique_exported_states": len({digest({"messages": p["visible_messages"],
                                                  "target": p["proposed_action"]}) for p in states}) == 100,
    }
    return {"checks": checks, "passed": all(checks.values()), "risks": dict(counts),
            "questions": len(selected), "states": len(states), "training_allowed": False}


def validate_training(base: list[dict], supplement: list[dict], source_ids: list[str],
                      internal_ids: set[str], internal_text: set[str], old_ids: set[str],
                      old_text: set[str]) -> dict:
    combined = base + supplement
    texts = [norm(question(row)) for row in combined]
    counts = Counter(action(row) for row in combined)
    supplemental_texts = {norm(question(row)) for row in supplement}
    if not all(texts):
        raise ValueError("empty_question_in_training")
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in combined:
        grouped[norm(question(row))].add("nonempty_search" if nonempty_search(row) else action(row))
    complete_pairs = sum({"answer", "nonempty_search"} <= kinds for kinds in grouped.values())
    checks = {
        "395_states": len(combined) == 395,
        "211_unique_questions": len(set(texts)) == 211,
        "100_complete_pairs": complete_pairs == 100,
        "25_new_distinct_questions": len(set(source_ids)) == len(source_ids) == 25 and
                                      len(supplemental_texts) == 25,
        "50_new_actions": len(supplement) == 50 and
                          Counter(action(row) for row in supplement) == {"search": 25, "answer": 25},
        "answer_45_to_55_percent": len(combined) > 0 and .45 <= counts["answer"] / len(combined) <= .55,
        "evidence_visible_search_at_least_40_percent": len(combined) > 0 and
                                                 sum(nonempty_search(row) for row in combined) / len(combined) >= .40,
        "no_evidence_search_at_most_10_percent": len(combined) > 0 and
                                          sum(action(row) == "search" and not nonempty_search(row)
                                              for row in combined) / len(combined) <= .10,
        "no_internal_overlap": not set(source_ids) & internal_ids and not supplemental_texts & internal_text,
        "no_historical_overlap": not set(source_ids) & old_ids and not supplemental_texts & old_text,
        "unique_states": len({digest(row) for row in combined}) == 395,
    }
    return {"checks": checks, "passed": all(checks.values()), "states": len(combined),
            "questions": len(set(texts)), "complete_pairs": complete_pairs,
            "actions": dict(counts), "training_allowed": False}


def source_manifest() -> dict:
    source = ROOT / "data/hotpot-full-train-v114"
    provenance = read(source / "download-provenance.json")
    files = []
    for entry in provenance["files"]:
        path = safe(ROOT / entry["path"], expect=source)
        if not re.fullmatch(r"train-\d+-of-\d+\.parquet", path.name):
            raise ValueError("non_training_parquet_in_provenance")
        if sha(path) != entry["sha256"]:
            raise ValueError("training_source_hash_mismatch")
        files.append(path)
    return {"files": files, "provenance_sha256": sha(source / "download-provenance.json")}


def baseline() -> dict:
    inputs = {
        "v117_train": ROOT / "data/evidence-sft-v117/frozen/train.jsonl",
        "v122_internal_packets": ROOT / "data/evidence-sft-v122-internal-candidates/review/internal-packets.jsonl",
        "v122_internal_corpus": ROOT / "data/evidence-sft-v122-internal-candidates/candidate-corpus-private.json",
        "v122_train_packets": ROOT / "data/evidence-sft-v122-train-candidates/review/internal-packets.jsonl",
        "v122_fixed_summary": ROOT / "runs/v122-download/v122-unified/fixed-summary.json",
        "v122_dual_audit": ROOT / "runs/v122-download/dual-audit-report.json",
        "v118_tasks_v101": ROOT / "backups/v118/extracted/runs/v118/eval/v101-tasks.json",
        "v118_tasks_v112": ROOT / "backups/v118/extracted/runs/v118/eval/v112-tasks.json",
        "v118_tasks_v118": ROOT / "backups/v118/extracted/runs/v118/eval/v118-tasks.json",
    }
    expected = {
        "v117_train": "26ced5bafb5210fefe9b007915598c8d58d2ebec156e938a48d86e1f1577ad8b",
        "v122_internal_corpus": "e43e7de50f043bd66d6859417ca3ae04564f4df14f0dfa5fce4611f4a3fbba70",
        "v122_internal_packets": "c52530b04b559a4d7ab3cd932c1ddd5dd1d1fa89b6a4c101b4d73ece53a06ac8",
    }
    hashes = {key: sha(path) for key, path in inputs.items()}
    checks = {key: hashes[key] == value for key, value in expected.items()}
    # Git state is metadata only: private data and prior untracked files stay separate.
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                              text=True).stdout.strip()
    source = source_manifest()
    has_git = (ROOT / ".git").exists()
    report = {"version": "v123-baseline-v1", "git_head": git("rev-parse", "HEAD") if has_git else None,
              "git_tree": git("rev-parse", "HEAD^{tree}") if has_git else None,
              "tracked_changes": git("diff", "--name-only") if has_git else None,
              "inputs_sha256": hashes, "source_provenance_sha256": source["provenance_sha256"],
              "source_files": [{"name": p.name, "sha256": sha(p)} for p in source["files"]],
              "checks": checks, "status": "passed" if all(checks.values()) else "failed",
              "gpu": "not_run", "model_inference": "not_run", "training_allowed": False}
    return report
