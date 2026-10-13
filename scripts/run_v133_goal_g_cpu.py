"""Bounded CPU-only Goal G locator, source screening, and tokenizer checks.

No model weights or inference. `locate` never reads answer/context columns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-g-cpu"
SOURCE = ROOT / "data/hotpot-full-train-v114"
PROTOCOL = OUT / "PROTOCOL.md"
SHARDS = ("train-00000-of-00002.parquet", "train-00001-of-00002.parquet")
LOCATOR = re.compile(r"\b(both|share|shared|common|same|each)\b", re.I)
SEED = "v133-goal-g-v1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def qhash(value: str) -> str:
    return hashlib.sha256(norm(value).encode()).hexdigest()


def readl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def writel(path: Path, values) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


EXPOSURE_FILES = {
    "v117_ids": "data/evidence-sft-v117/candidate-freeze.json",
    "v117_cases": "data/evidence-sft-v117/candidate-private.jsonl",
    "v118_actual_train": "data/evidence-sft-v117/frozen/train.jsonl",
    "v122_train_history": "data/evidence-sft-v122-train-candidates/history-private.json",
    "v122_internal_history": "data/evidence-sft-v122-internal-candidates/history-private.json",
    "v122_train_ids": "data/evidence-sft-v122-train-candidates/candidate-freeze.json",
    "v122_internal_ids": "data/evidence-sft-v122-internal-candidates/candidate-freeze.json",
    "v122_train_cases": "data/evidence-sft-v122-train-candidates/candidate-private.jsonl",
    "v122_internal_cases": "data/evidence-sft-v122-internal-candidates/candidate-private.jsonl",
    "v123_screen": "data/evidence-sft-v123/internal/screened-source-private.jsonl",
    "v130_screen": "data/evidence-sft-v124-forensic/goal-v128/g4-new-pool/screened-source-private.jsonl",
    "goal_c_screen": "data/evidence-sft-v133-goal-c-private/screen-ledger-private.jsonl",
    "goal_c_map": "data/evidence-sft-v133-goal-c-private/source-map-private.jsonl",
    "goal_c_train_index": "data/evidence-sft-v133-goal-c-private/frozen/train-review-index-private.jsonl",
    "goal_c_dev_index": "data/evidence-sft-v133-goal-c-private/frozen/dev-source-index-private.jsonl",
    "goal_c_train": "data/evidence-sft-v133-goal-c-private/frozen/train-candidate.jsonl",
    "goal_c_dev": "data/evidence-sft-v133-goal-c-private/frozen/dev-candidate.jsonl",
    "v118_config": "configs/v118-sft.json",
}


def source_files():
    provenance = json.loads((SOURCE / "download-provenance.json").read_text(encoding="utf-8"))
    if (provenance.get("dataset"), provenance.get("config"), provenance.get("split"), provenance.get("rows")) != (
        "hotpotqa/hotpot_qa", "distractor", "train", 90447
    ):
        raise ValueError("source_provenance_mismatch")
    expected = {Path(x["path"]).name: x for x in provenance["files"]}
    if set(expected) != set(SHARDS):
        raise ValueError("unexpected_source_shards")
    files = []
    for name in SHARDS:
        path = SOURCE / name
        if sha(path) != expected[name]["sha256"] or pq.ParquetFile(path).metadata.num_rows != expected[name]["rows"]:
            raise ValueError(f"source_hash_or_row_count_mismatch:{name}")
        files.append(path)
    return files


def exposure():
    ids, questions, titles = set(), set(), set()
    inputs = {}
    for key, relative in EXPOSURE_FILES.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        inputs[relative] = sha(path)
        if key.endswith("_ids"):
            ids.update(json.loads(path.read_text(encoding="utf-8"))["ids"])
        elif key.endswith("_history"):
            value = json.loads(path.read_text(encoding="utf-8"))
            ids.update(value["ids"])
            questions.update(qhash(x) for x in value["normalized_questions"])
        elif key.endswith("_cases"):
            for row in readl(path):
                packet = row.get("packet", {})
                if packet.get("question"):
                    questions.add(qhash(packet["question"]))
        elif key == "v118_actual_train" or key in ("goal_c_train", "goal_c_dev"):
            for row in readl(path):
                messages = row["messages"]
                questions.update(qhash(m["content"]) for m in messages if m["role"] == "user")
                for m in messages:
                    if m["role"] == "tool":
                        observation = json.loads(m["content"])
                        titles.update(norm(x["title"]) for x in observation.get("result", []) if x.get("title"))
        elif key in ("v123_screen", "v130_screen", "goal_c_screen"):
            for row in readl(path):
                if row.get("id") or row.get("source_id"):
                    ids.add(row.get("id") or row.get("source_id"))
                if row.get("question_sha256"):
                    questions.add(row["question_sha256"])
        elif key in ("goal_c_map", "goal_c_train_index", "goal_c_dev_index"):
            for row in readl(path):
                if row.get("source_id"):
                    ids.add(row["source_id"])
                if row.get("question_sha256"):
                    questions.add(row["question_sha256"])
        elif key == "v118_config":
            value = json.loads(path.read_text(encoding="utf-8"))
            if value["train_data"] != EXPOSURE_FILES["v118_actual_train"]:
                raise ValueError("v118_train_manifest_mismatch")
    return ids, questions, titles, inputs


def locate():
    if (OUT / "locator.jsonl").exists():
        raise FileExistsError("locator_already_frozen")
    files = source_files()
    ids, questions, titles, inputs = exposure()
    entries, counts = [], Counter()
    start = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["id", "question", "type"], batch_size=4096):
            for row in batch.to_pylist():
                index = start
                start += 1
                counts["metadata_rows"] += 1
                question = str(row.get("question") or "")
                if not LOCATOR.search(question):
                    continue
                counts["locator_matches"] += 1
                source_id = str(row["id"])
                question_sha = qhash(question)
                if source_id in ids or question_sha in questions:
                    counts["exposed_before_full_open"] += 1
                    continue
                entries.append({"source_id": source_id, "source_row_index": index,
                                "question": question, "question_sha256": question_sha,
                                "type": row.get("type"), "sort_key": hashlib.sha256((SEED + source_id).encode()).hexdigest()})
    if start != 90447:
        raise ValueError("metadata_row_count_mismatch")
    entries.sort(key=lambda x: x["sort_key"])
    writel(OUT / "locator.jsonl", entries)
    write(OUT / "locator-summary.json", {"status": "frozen", "protocol_sha256": sha(PROTOCOL),
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "provenance_sha256": sha(SOURCE / "download-provenance.json"),
        "input_sha256": inputs, "exposure_ids": len(ids), "exposure_questions": len(questions),
        "training_dev_doc_titles": len(titles), "counts": dict(counts),
        "eligible_metadata_count": len(entries), "locator_sha256": sha(OUT / "locator.jsonl")})
    print(json.dumps({"counts": counts, "eligible_metadata_count": len(entries)}))


def official_facts(row):
    facts = row.get("supporting_facts") or {}
    context = row.get("context") or {}
    title_list = context.get("title") or []
    sentence_list = context.get("sentences") or []
    selected = []
    for title, sentence_id in zip(facts.get("title") or [], facts.get("sent_id") or []):
        try:
            position = title_list.index(title)
            sentence = sentence_list[position][sentence_id]
            if not isinstance(sentence, str) or not sentence.strip():
                return None
        except (ValueError, IndexError, TypeError):
            return None
        selected.append({"title": title, "sentence_index": int(sentence_id), "text": sentence.strip()})
    if len(selected) != len(facts.get("title") or []):
        return None
    return selected


def screen():
    summary = json.loads((OUT / "locator-summary.json").read_text(encoding="utf-8"))
    if summary["protocol_sha256"] != sha(PROTOCOL) or summary["locator_sha256"] != sha(OUT / "locator.jsonl"):
        raise ValueError("locator_or_protocol_changed")
    if (OUT / "screen-ledger.jsonl").exists():
        raise FileExistsError("screen_already_frozen")
    files = source_files()
    locators = list(readl(OUT / "locator.jsonl"))[:100]
    positions = {x["source_row_index"]: x for x in locators}
    selected = {}
    start = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for gi in range(pf.metadata.num_row_groups):
            size = pf.metadata.row_group(gi).num_rows
            wanted = [i for i in positions if start <= i < start + size]
            if wanted:
                table = pf.read_row_group(gi, columns=["id", "answer", "supporting_facts", "context"])
                for index in wanted:
                    offset = index - start
                    selected[index] = {key: table[key][offset].as_py() for key in table.column_names}
            start += size
    if set(selected) != set(positions):
        raise ValueError("opened_row_mismatch")
    _, _, exposed_titles, _ = exposure()
    ledger, provisional = [], []
    for locator in locators:
        index = locator["source_row_index"]
        row = selected[index]
        if str(row["id"]) != locator["source_id"]:
            raise ValueError("source_id_changed")
        answer = str(row.get("answer") or "").strip()
        facts = official_facts(row)
        titles = list(dict.fromkeys(x["title"] for x in facts)) if facts else []
        reason = None
        if not answer or norm(answer) in ("yes", "no"):
            reason = "boolean_or_empty_answer"
        elif not facts or len(titles) != 2:
            reason = "not_two_resolved_evidence_titles"
        elif any(norm(t) in exposed_titles for t in titles):
            reason = "evidence_title_overlaps_known_training_or_dev"
        item = {**locator, "answer": answer, "supporting_facts": facts,
                "evidence_titles": titles, "screen_reason": reason or "needs_semantic_and_entity_review"}
        ledger.append(item)
        if not reason:
            provisional.append(item)
    writel(OUT / "screen-ledger.jsonl", ledger)
    writel(OUT / "provisional.jsonl", provisional)
    write(OUT / "screen-summary.json", {"opened_full_rows": len(ledger),
        "structural_candidates": len(provisional), "reason_counts": dict(Counter(x["screen_reason"] for x in ledger)),
        "ledger_sha256": sha(OUT / "screen-ledger.jsonl"), "provisional_sha256": sha(OUT / "provisional.jsonl")})
    print(json.dumps({"opened": len(ledger), "structural_candidates": len(provisional),
                      "reason_counts": dict(Counter(x["screen_reason"] for x in ledger))}))


# Reviewed manually against the complete official supporting sentences in the
# frozen provisional order. These are screening judgments, never reviewer votes.
MANUAL_ACCEPT = {0, 1, 2, 5, 6, 8, 11, 13, 16, 21, 24, 29, 36, 37, 38, 39, 43, 45, 48}
MANUAL_REJECT = {
    3: "shared_project_not_shared_category", 4: "single_entity_lookup",
    7: "date_question", 9: "official_answer_not_joint_category",
    10: "single_entity_relation", 12: "which_case_comparison",
    14: "proposed_shared_attribute_false", 15: "relation_chain_not_shared_category",
    17: "person_and_band_have_different_roles", 18: "single_entity_lookup",
    19: "date_question", 20: "one_support_title_unrelated_to_joint_team",
    22: "second_evidence_does_not_state_research_university",
    23: "manager_lookup", 25: "produced_versus_released_company_conflict",
    26: "date_question", 27: "single_person_lookup", 28: "shared_cast_lookup",
    30: "creator_lookup", 31: "shared_film_lookup",
    32: "sport_question_but_official_answer_names_person_category",
    33: "museum_has_no_ethnicity", 34: "show_lookup", 35: "founder_lookup",
    40: "creator_vs_user_of_system", 41: "character_lookup", 42: "prize_recipient_lookup",
    44: "first_document_does_not_establish_American_show",
    46: "single_person_nationality", 47: "film_lookup", 49: "conditional_yes_no_question",
}


def adjudicate():
    if (OUT / "semantic-ledger.jsonl").exists():
        raise FileExistsError("semantic_screen_already_frozen")
    expected = json.loads((OUT / "screen-summary.json").read_text(encoding="utf-8"))
    if expected["provisional_sha256"] != sha(OUT / "provisional.jsonl"):
        raise ValueError("provisional_changed")
    cases = list(readl(OUT / "provisional.jsonl"))
    if len(cases) != 50 or MANUAL_ACCEPT & MANUAL_REJECT.keys() or MANUAL_ACCEPT | MANUAL_REJECT.keys() != set(range(50)):
        raise ValueError("manual_disposition_incomplete_or_order_changed")
    train = list(readl(ROOT / EXPOSURE_FILES["v118_actual_train"]))
    train += list(readl(ROOT / EXPOSURE_FILES["goal_c_train"]))
    train += list(readl(ROOT / EXPOSURE_FILES["goal_c_dev"]))
    exposed_questions = [norm(m["content"]) for row in train for m in row["messages"] if m["role"] == "user"]
    ledger, review, entity_keys = [], [], set()
    for i, item in enumerate(cases):
        names = [norm(t) for t in item["evidence_titles"]]
        overlap = [name for name in names if any(name in q for q in exposed_questions)]
        selected_overlap = [name for name in names if name in entity_keys]
        reason = MANUAL_REJECT.get(i)
        if i in MANUAL_ACCEPT and (overlap or selected_overlap):
            reason = "named_entity_overlap_with_actual_train_dev_or_selected_case"
        status = "ready_for_anonymous_review" if i in MANUAL_ACCEPT and not reason else "excluded"
        record = {"provisional_index": i, "source_id": item["source_id"],
                  "question_sha256": item["question_sha256"], "status": status,
                  "reason": reason, "evidence_titles": item["evidence_titles"],
                  "entity_overlap_with_actual_train_dev": overlap,
                  "entity_overlap_with_selected": selected_overlap}
        ledger.append(record)
        if status == "ready_for_anonymous_review":
            entity_keys.update(names)
            review.append(item)
    writel(OUT / "semantic-ledger.jsonl", ledger)
    writel(OUT / "review-candidates.jsonl", review)
    write(OUT / "semantic-summary.json", {"screened": len(cases), "ready_for_review": len(review),
        "excluded": len(cases)-len(review), "manual_candidate_indices": sorted(MANUAL_ACCEPT),
        "entity_check": "exact normalized official title in actual v118/Goal C train/dev questions plus selected-case titles; manual semantic names inspected",
        "semantic_ledger_sha256": sha(OUT / "semantic-ledger.jsonl"),
        "review_candidates_sha256": sha(OUT / "review-candidates.jsonl")})
    print(json.dumps({"screened": len(cases), "ready_for_review": len(review)}))


SYSTEM = "You are an evidence-driven retrieval agent. Output one JSON search or answer action, using visible evidence only."


def make_case(item):
    """Build the answer-blind Goal E style history from exact official facts."""
    facts = item["supporting_facts"]
    titles = item["evidence_titles"]
    if not facts or len(titles) != 2:
        raise ValueError("not_two_evidence_titles")
    documents = []
    for title in titles:
        content = " ".join(x["text"] for x in facts if x["title"] == title)
        digest = hashlib.sha256((title + "\n" + content).encode()).hexdigest()
        documents.append({"id": digest[:20], "title": title, "text": content,
                          "source": "official-hotpotqa-train-supporting-fact:" + digest[:12]})
    def tool(document):
        return json.dumps({"ok": True, "result": [document]}, ensure_ascii=False, separators=(",", ":"))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": json.dumps({"action": "search", "query": item["question"]}, ensure_ascii=False)},
        {"role": "tool", "content": tool(documents[0])},
        {"role": "assistant", "content": json.dumps({"action": "search", "query": documents[0]["text"]}, ensure_ascii=False)},
        {"role": "tool", "content": tool(documents[1])},
        {"role": "assistant", "content": json.dumps({"action": "answer", "answer": item["answer"]}, ensure_ascii=False)},
    ]
    return {"messages": messages, "target_assistant_index": 6, "thinking_enabled": False,
            "source_id": item["source_id"], "question_sha256": item["question_sha256"],
            "evidence_titles": titles, "evidence_documents": documents,
            "official_answer": item["answer"]}


def cpu_check(processor, row):
    from mini_agent_rl.training.sft import encode_assistant_only, AssistantOnlyCollator

    messages = row["messages"]
    target = row["target_assistant_index"]
    if target != len(messages) - 1 or messages[target]["role"] != "assistant":
        raise ValueError("wrong_target_assistant")
    if [m["role"] for m in messages] != ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]:
        raise ValueError("wrong_history_roles")
    expected = json.loads(messages[target]["content"])
    if expected != {"action": "answer", "answer": row["official_answer"]}:
        raise ValueError("answer_target_mismatch")
    observations = [json.loads(messages[i]["content"]) for i in (3, 5)]
    if any(x.get("ok") is not True or len(x.get("result", [])) != 1 for x in observations):
        raise ValueError("missing_one_of_two_evidence_documents")
    if [x["result"][0] for x in observations] != row["evidence_documents"]:
        raise ValueError("rendered_evidence_binding_mismatch")
    full = encode_assistant_only(processor, messages, 2176, target, False)
    ids, labels = full["input_ids"], full["labels"]
    prompt = processor.apply_chat_template(messages[:target], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    if isinstance(prompt, dict):
        prompt = prompt["input_ids"]
    prompt = list(prompt[0] if prompt and isinstance(prompt[0], list) else prompt)
    positions = [i for i, x in enumerate(labels) if x != -100]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError("noncontiguous_supervision")
    if ids[:positions[0]] != prompt or any(labels[i] != ids[i] for i in positions):
        raise ValueError("prefix_or_label_mismatch")
    if positions[0] == 0 or ids[positions[-1]] != processor.tokenizer.convert_tokens_to_ids("<|im_end|>"):
        raise ValueError("causal_shift_or_terminator_error")
    if len(prompt) > 2048 or len(positions) > 128 or len(ids) > 2176:
        raise ValueError("token_limit_exceeded")
    rendered = processor.apply_chat_template(messages[:target], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    if messages[target]["content"] in rendered:
        raise ValueError("target_message_leaked")
    if messages[1]["content"] not in rendered or any(m["content"] not in rendered for m in messages if m["role"] == "tool"):
        raise ValueError("question_or_evidence_missing")
    # Check the actual collator's right-padding behavior on unequal lengths.
    collator = AssistantOnlyCollator(processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
    padded = collator([full, {"input_ids": ids[:-1], "attention_mask": full["attention_mask"][:-1], "labels": labels[:-1]}])
    if padded["attention_mask"][1, -1].item() != 0 or padded["labels"][1, -1].item() != -100:
        raise ValueError("padding_mask_error")
    content = messages[target]["content"]
    encoded_answer = json.dumps(row["official_answer"], ensure_ascii=False)
    # JSON quote marks are format tokens. Offsets for the answer value are the
    # escaped serialized content strictly inside those marks.
    start = content.index(encoded_answer, content.index('"answer"')) + 1
    end = start + len(encoded_answer) - 2
    offsets = processor.tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    content_ids = processor.tokenizer(content, add_special_tokens=False)["input_ids"]
    if content_ids != [ids[i] for i in positions[:-1]]:
        raise ValueError("content_tokenization_mismatch")
    answer_tokens, format_tokens, ambiguous = classify_offsets(offsets, start, end)
    format_tokens += 1  # assistant terminator
    if ambiguous:
        raise ValueError(f"answer_format_token_boundary_ambiguous:{ambiguous}")
    return {"prompt_tokens": len(prompt), "target_tokens": len(positions), "total_tokens": len(ids),
            "answer_tokens": answer_tokens, "format_tokens": format_tokens,
            "prompt_sha256": hashlib.sha256(bytes().join(int(i).to_bytes(4, "little") for i in prompt)).hexdigest()}


def classify_offsets(offsets, start, end):
    answer_tokens = format_tokens = ambiguous = 0
    for lo, hi in offsets:
        if lo >= start and hi <= end:
            answer_tokens += 1
        elif hi <= start or lo >= end:
            format_tokens += 1
        else:
            ambiguous += 1
    return answer_tokens, format_tokens, ambiguous


def preflight():
    from transformers import AutoProcessor
    import torch
    import importlib.metadata as metadata

    if torch.cuda.is_available():
        raise RuntimeError("CPU_only_required")
    result_path = OUT / "cpu-preflight-v2.jsonl"
    summary_path = OUT / "cpu-summary-v2.json"
    if result_path.exists():
        raise FileExistsError("cpu_preflight_v2_already_frozen")
    prior = json.loads((OUT / "cpu-summary.json").read_text(encoding="utf-8"))
    if prior.get("cases") != 19 or prior.get("passed") != 0:
        raise ValueError("unexpected_prior_failure_set")
    source = list(readl(OUT / "review-candidates.jsonl"))
    processor = AutoProcessor.from_pretrained("/root/autodl-tmp/qwen_08b", local_files_only=True)
    results = []
    for item in source:
        try:
            row = make_case(item)
            check = cpu_check(processor, row)
            results.append({"source_id": item["source_id"], "status": "pass", "check": check})
        except Exception as exc:
            results.append({"source_id": item["source_id"], "status": "fail", "reason": str(exc),
                            "error_type": type(exc).__name__})
    writel(result_path, results)
    write(summary_path, {"cases": len(results), "passed": sum(x["status"] == "pass" for x in results),
          "versions": {name: metadata.version(name) for name in ("torch", "transformers", "tokenizers")},
          "tokenizer_dir": "/root/autodl-tmp/qwen_08b", "gpu_used": False,
          "prior_results_sha256": sha(OUT / "cpu-preflight.jsonl"), "repair": "exclude JSON quote tokens from answer value span",
          "results_sha256": sha(result_path)})
    print(json.dumps({"cases": len(results), "passed": sum(x["status"] == "pass" for x in results)}))


def review_packets():
    path = OUT / "review-packets.jsonl"
    if path.exists():
        raise FileExistsError("review_packets_already_frozen")
    cases = list(readl(OUT / "review-candidates.jsonl"))
    checks = {x["source_id"]: x for x in readl(OUT / "cpu-preflight-v2.jsonl")}
    packets = []
    for i, item in enumerate(cases):
        if checks[item["source_id"]]["status"] != "pass":
            continue
        case = make_case(item)
        packets.append({"case_id": f"g{i:03d}", "question": item["question"],
            "visible_messages": case["messages"][:6],
            "before_action": case["evidence_documents"], "retrieved_after_search": [],
            "proposed_action": {"action": "answer", "answer": item["answer"]},
            "review_scope": "Check open-ended common category/property of two entities, same answer granularity, direct support separately in both visible documents, ambiguity and answer leakage. A mere answer-string match is insufficient.",
            "protocol": {"version": "v130-evidence-review-citations-v1"}})
    writel(path, packets)
    write(OUT / "review-packets-summary.json", {"count": len(packets), "packet_sha256": sha(path),
        "source_case_map_sha256": sha(OUT / "review-candidates.jsonl"),
        "cpu_preflight_sha256": sha(OUT / "cpu-preflight-v2.jsonl"),
        "reviewer_visibility": "anonymous identical packet; no source ID, split, performance, or other vote"})
    print(json.dumps({"packets": len(packets), "sha256": sha(path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("locate", "screen", "adjudicate", "preflight", "review-packets"))
    args = parser.parse_args()
    {"locate": locate, "screen": screen, "adjudicate": adjudicate, "preflight": preflight,
     "review-packets": review_packets}[args.phase]()
