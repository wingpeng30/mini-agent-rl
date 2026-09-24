"""Freeze unselected train-source candidate states for bounded follow-up review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transformers import AutoProcessor

from build_v114_sft_candidates import public_example
from prepare_v114_reviews import ROOT, SOURCE, MODEL, canonical, freeze, sha
from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.domain import Message
from mini_agent_rl.model.context import ContextBudgetError, budget_messages
from mini_agent_rl.training.sft import encode_assistant_only

OUT = ROOT / "data/evidence-sft-v114-review"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    for filename in ("candidate-cache-training-private.jsonl", "candidate-cache-internal-private.jsonl",
                     "blind-review-candidates.jsonl", "internal-freeze.json", "training-question-freeze.json"):
        if sha((SOURCE / filename).read_bytes()) != manifest["files"][filename]:
            raise RuntimeError(f"source_hash_mismatch:{filename}")
    original = {row["case_id"] for row in load(SOURCE / "blind-review-candidates.jsonl")}
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)

    def render(messages: list[Message]) -> list[int]:
        value = processor.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        if isinstance(value, dict):
            value = value["input_ids"]
        return value[0] if value and isinstance(value[0], list) else value

    rows = []
    private = []
    failures = []
    for split, filename in (("train", "candidate-cache-training-private.jsonl"),
                            ("internal", "candidate-cache-internal-private.jsonl")):
        for candidate in load(SOURCE / filename):
            public = public_example(candidate, 0)
            case_id = public["case_id"]
            if case_id in original:
                continue
            try:
                original_messages = [Message(**m) for m in public["visible_messages"]]
                budgeted = budget_messages(original_messages, 2048, lambda ms: len(render(ms)),
                                           lambda s: len(processor.tokenizer.encode(s, add_special_tokens=False)),
                                           observation_encoding="compact-v3")
                actual = [{"role": m.role, "content": m.content} for m in budgeted.messages]
                ids = render(budgeted.messages)
                target = {"role": "assistant", "content": json.dumps(public["proposed_action"], ensure_ascii=False, separators=(",", ":"))}
                encoded = encode_assistant_only(processor, actual + [target], 2176, len(actual), False)
                target_tokens = sum(x != -100 for x in encoded["labels"])
                if target_tokens > 128:
                    raise ValueError("target_exceeds_128")
                packet = {"case_id": case_id, "question": public["question"],
                          "visible_messages": actual, "proposed_action": public["proposed_action"],
                          "visible_evidence": visible_documents(actual), "retrieval_top3": public["retrieval_top3"],
                          "prompt_sha256": sha(bytes(str(ids), "utf-8")),
                          "prompt_tokens": len(ids), "target_tokens": target_tokens,
                          "preflight_error": None}
                packet["packet_sha256"] = sha(canonical(packet))
                rows.append(packet)
                private.append({"case_id": case_id, "source_id": candidate["qid"], "split": split,
                                "state_kind": candidate["state_kind"], "risk_tags": candidate["risk_tags"]})
            except (ContextBudgetError, ValueError, KeyError, TypeError) as exc:
                failures.append({"case_id": case_id, "reason": f"{type(exc).__name__}:{exc}"})
    if len({r["case_id"] for r in rows}) != len(rows):
        raise RuntimeError("duplicate_reserve_case_id")
    # Deterministic order prioritizes rare risk categories and question diversity.
    by_id = {r["case_id"]: r for r in rows}
    private.sort(key=lambda r: (r["split"] != "internal", "measurement" not in r["risk_tags"],
                                "time" not in r["risk_tags"],
                                sha(("v114-reserve:" + r["source_id"]).encode()), r["case_id"]))
    ordered = [by_id[r["case_id"]] for r in private]
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / "reserve-private.jsonl", b"".join(canonical(r) + b"\n" for r in private))
    for reviewer in ("a", "b"):
        freeze(OUT / f"reserve-{reviewer}-packets.jsonl", b"".join(canonical(r) + b"\n" for r in ordered))
    freeze(OUT / "reserve-inventory.json", json.dumps({
        "source_manifest_sha256": sha((SOURCE / "manifest.json").read_bytes()),
        "original_candidates": 419, "reserve_candidates": len(rows),
        "reserve_by_split": {split: sum(r["split"] == split for r in private) for split in ("train", "internal")},
        "preflight_failures": failures,
        "files_sha256": {name: sha((OUT / name).read_bytes()) for name in
                         ("reserve-private.jsonl", "reserve-a-packets.jsonl", "reserve-b-packets.jsonl")},
    }, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"reserves": len(rows), "preflight_failures": len(failures),
                      "by_split": {split: sum(r["split"] == split for r in private) for split in ("train", "internal")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
