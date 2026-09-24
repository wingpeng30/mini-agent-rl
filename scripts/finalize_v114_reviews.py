"""Fail-closed, resumable join of two blinded Luna review passes."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/evidence-sft-v114-clean"
WORK = ROOT / "data/evidence-sft-v114-review"
REPORT = ROOT / "reports/v114-review-gate.json"
RISKS = {"relation", "measurement", "time", "scope"}
ALLOWED_REVIEW_RISKS = RISKS | {"entity"}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_luna_reviews(reviewer: str, stem: str, packets: dict[str, dict],
                           expected_order: list[str]) -> Path:
    """Translate the audit agents' preserved schemas into the strict gate schema."""
    raw_path = WORK / f"{stem}-{reviewer}-decisions.jsonl"
    addenda_path = WORK / f"{stem}-{reviewer}-addenda.jsonl"
    raw_rows = load(raw_path)
    if [row.get("case_id") for row in raw_rows] != expected_order[:len(raw_rows)]:
        # Keep load_decisions' canonical order error, and do not silently reorder audit output.
        return raw_path
    addenda_rows = load(addenda_path)
    addenda = {row.get("case_id"): row for row in addenda_rows}
    approved_ids = []
    normalized = []
    for row in raw_rows:
        case_id = row.get("case_id")
        packet = packets.get(case_id)
        verdict = row.get("decision", row.get("verdict"))
        if verdict not in {"approve", "approved", "reject", "rejected"}:
            approved = None
        else:
            approved = verdict in {"approve", "approved"}
        if approved:
            approved_ids.append(case_id)
        citations = []
        visible = {d.get("id"): d for d in packet.get("visible_evidence", [])} if packet else {}
        retrieved = {d.get("id"): d for d in packet.get("retrieval_top3", [])} if packet else {}
        for citation in row.get("citations", []) if isinstance(row.get("citations"), list) else []:
            document_id = citation.get("document_id", citation.get("evidence_id"))
            doc = visible.get(document_id) or retrieved.get(document_id)
            stage = citation.get("stage")
            if stage in {"visible_evidence", "visible"}:
                stage = "visible"
            elif stage in {"retrieved", "retrieval_top3"}:
                stage = "retrieved"
            elif stage is None:
                stage = "visible" if document_id in visible else "retrieved" if document_id in retrieved else None
            title = citation.get("title")
            # An explicitly supplied title must match the packet document bound to its id.
            if title is not None and (doc is None or title != doc.get("title")):
                stage = None
            citations.append({"stage": stage, "document_id": document_id,
                              "quote": citation.get("quote")})
        supplemental = addenda.get(case_id, {})
        if supplemental.get("packet_sha256") != (packet or {}).get("packet_sha256"):
            supplemental = {}
        normalized.append({
            "case_id": case_id,
            "packet_sha256": row.get("packet_sha256"),
            "reviewer_id": f"luna-{reviewer}",
            "approved": approved,
            "reason": row.get("reason"),
            "issue_tags": row.get("issue_tags") if isinstance(row.get("issue_tags"), list) else [],
            "risk_labels": supplemental.get("risk_labels"),
            "citations": citations,
            "missing_fact": supplemental.get("missing_fact"),
        })
    # The addendum must cover exactly the approvals, in their original packet order.
    if [row.get("case_id") for row in addenda_rows] != approved_ids:
        for row in normalized:
            if row["approved"]:
                row["risk_labels"] = None
                if packets[row["case_id"]]["proposed_action"].get("action") == "search":
                    row["missing_fact"] = None
    output = WORK / f"{stem}-{reviewer}-normalized-decisions.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                                 for row in normalized), encoding="utf-8")
    return output


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def norm(value: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in unicodedata.normalize("NFKC", value).casefold()).split())


def normalized_risks(labels) -> set[str]:
    return {"relation" if label == "entity" else label for label in labels}


def docs_by_stage(packet: dict) -> dict[str, dict[str, str]]:
    result = {}
    for d in packet["visible_evidence"]:
        if d.get("id") and d.get("text"):
            result[("visible", d["id"])] = d
    for d in packet["retrieval_top3"]:
        if d.get("id") and d.get("text"):
            result[("retrieved", d["id"])] = d
    return result


def valid_review(review: dict, packet: dict, reviewer: str) -> list[str]:
    errors = []
    if review.get("reviewer_id") != f"luna-{reviewer}":
        errors.append("reviewer_id_mismatch")
    if review.get("packet_sha256") != packet["packet_sha256"]:
        errors.append("packet_sha256_mismatch")
    if type(review.get("approved")) is not bool:
        errors.append("missing_boolean_approval")
    if not isinstance(review.get("reason"), str) or not review["reason"].strip():
        errors.append("missing_reason")
    if not isinstance(review.get("issue_tags"), list):
        errors.append("invalid_issue_tags")
    if not review.get("approved"):
        return errors
    risks = review.get("risk_labels")
    if not isinstance(risks, list) or any(x not in ALLOWED_REVIEW_RISKS for x in risks):
        errors.append("invalid_risk_labels")
    if packet.get("preflight_error"):
        errors.append("preflight_failed")
    citations = review.get("citations")
    if not isinstance(citations, list) or not citations:
        return errors + ["missing_citations"]
    available = docs_by_stage(packet)
    stages = set()
    new_cited = False
    visible_ids = {d["id"] for d in packet["visible_evidence"]}
    for citation in citations:
        if not isinstance(citation, dict):
            errors.append("malformed_citation")
            continue
        stage, doc_id, quote = citation.get("stage"), citation.get("document_id"), citation.get("quote")
        doc = available.get((stage, doc_id))
        if doc is None or not isinstance(quote, str) or not quote.strip() or quote not in doc["text"]:
            errors.append("illegal_verbatim_citation")
        else:
            stages.add(stage)
            if stage == "retrieved" and doc_id not in visible_ids:
                new_cited = True
    action = packet["proposed_action"]["action"]
    if action == "answer":
        if stages != {"visible"}:
            errors.append("answer_requires_visible_only_citations")
    elif action == "search":
        if not new_cited:
            errors.append("search_requires_new_top3_citation")
        if not isinstance(review.get("missing_fact"), str) or not review["missing_fact"].strip():
            errors.append("search_requires_missing_fact")
    else:
        errors.append("unknown_action")
    return errors


def load_decisions(name: str, packets: dict[str, dict], path: Path | None = None,
                   expected_order: list[str] | None = None) -> tuple[dict[str, dict], list[dict]]:
    decisions = {}
    errors = []
    duplicate_ids = set()
    rows = load(path or WORK / f"reviewer-{name}-decisions.jsonl")
    if expected_order is not None and [row.get("case_id") for row in rows] != expected_order[:len(rows)]:
        errors.append({"case_id": None, "reviewer": name, "error": "decision_order_mismatch"})
    for review in rows:
        case_id = review.get("case_id")
        if case_id in decisions:
            errors.append({"case_id": case_id, "reviewer": name, "error": "duplicate_decision"})
            duplicate_ids.add(case_id)
            continue
        packet = packets.get(case_id)
        if packet is None:
            errors.append({"case_id": case_id, "reviewer": name, "error": "foreign_case"})
            continue
        validation = valid_review(review, packet, name)
        if validation:
            errors.append({"case_id": case_id, "reviewer": name, "error": validation})
        decisions[case_id] = {**review, "_invalid_errors": validation}
    for case_id in duplicate_ids:
        decisions[case_id]["_invalid_errors"].append("duplicate_decision")
    # A sentence reused across unrelated questions is generic; repeated cases
    # for the same question may share a concise rationale when each citation
    # and packet has still been audited independently.
    reason_questions: dict[str, set[str]] = {}
    for case_id, review in decisions.items():
        if review.get("approved"):
            reason_questions.setdefault(review.get("reason", "").strip(), set()).add(norm(packets[case_id]["question"]))
    for case_id, review in decisions.items():
        if review.get("approved") and len(reason_questions.get(review.get("reason", "").strip(), set())) > 1:
            review["_invalid_errors"].append("non_specific_repeated_approval_reason")
            errors.append({"case_id": case_id, "reviewer": name, "error": ["non_specific_repeated_approval_reason"]})
    return decisions, errors


def apply_battlefield_rereview(base_a: dict[str, dict], errors_a: list[dict],
                               packets: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Accept one append-only semantic re-review without editing its bad original row."""
    path = WORK / "reviewer-a-battlefield-append-only-rereview.jsonl"
    rows = load(path)
    if not rows:
        return [], errors_a
    if len(rows) != 1:
        raise RuntimeError("battlefield_rereview_count_invalid")
    row = rows[0]
    case_id = row.get("case_id")
    prior = base_a.get(case_id)
    packet = packets.get(case_id)
    if prior is None or packet is None or row.get("reviewer_id") != "luna-a":
        raise RuntimeError("battlefield_rereview_case_invalid")
    raw_old = next((line.strip() for line in (WORK / "reviewer-a-decisions.jsonl").read_text(encoding="utf-8").splitlines()
                    if json.loads(line).get("case_id") == case_id), None)
    raw_packet = next((line.strip() for line in (WORK / "reviewer-a-packets.jsonl").read_text(encoding="utf-8").splitlines()
                       if json.loads(line).get("case_id") == case_id), None)
    if raw_old is None or raw_packet is None:
        raise RuntimeError("battlefield_rereview_source_row_missing")
    old_hash = hashlib.sha256(raw_old.encode("utf-8")).hexdigest()
    packet_record_hash = hashlib.sha256(raw_packet.encode("utf-8")).hexdigest()
    if (row.get("packet_sha256") != packet.get("packet_sha256") or
            row.get("packet_record_sha256") != packet_record_hash or
            row.get("supersedes", {}).get("decision_record_sha256") != old_hash or
            row.get("supersedes", {}).get("decision_file") != "data\\evidence-sft-v114-review\\reviewer-a-decisions.jsonl" or
            prior.get("approved") is not True or row.get("approved") is not True):
        raise RuntimeError("battlefield_rereview_supersession_mismatch")
    clean = {key: row.get(key) for key in
             ("case_id", "packet_sha256", "reviewer_id", "approved", "reason", "issue_tags",
              "risk_labels", "citations", "missing_fact")}
    if clean["issue_tags"] is None:
        clean["issue_tags"] = []
    clean["citations"] = [
        {**citation, "stage": "visible" if citation.get("stage") == "visible_evidence" else citation.get("stage")}
        for citation in (clean["citations"] or [])
    ]
    validation = valid_review(clean, packet, "a")
    if validation:
        raise RuntimeError("battlefield_rereview_invalid:" + ",".join(validation))
    prior_errors = prior.get("_invalid_errors", [])
    if "missing_citations" not in prior_errors:
        raise RuntimeError("battlefield_rereview_did_not_repair_expected_error")
    base_a[case_id] = {**clean, "_invalid_errors": []}
    filtered = [item for item in errors_a if not (
        item.get("case_id") == case_id and item.get("reviewer") == "a" and
        item.get("error") == ["missing_citations"])]
    return [{"case_id": case_id, "decision_record_sha256": old_hash,
             "rereview_record_sha256": digest(row), "packet_record_sha256": packet_record_hash,
             "status": "accepted_append_only_re_review"}], filtered


def load_measurement_risk_recheck(packets: dict[str, dict]) -> tuple[set[str], dict]:
    packet_path = WORK / "measurement-risk-recheck-packets.jsonl"
    raw_packets = load(packet_path)
    if not raw_packets:
        return set(), {"candidate_cases": 0, "reviewer_a_true": 0, "reviewer_b_true": 0,
                       "dual_true": 0, "applied_cases": []}
    ids = [row.get("case_id") for row in raw_packets]
    if len(ids) != len(set(ids)):
        raise RuntimeError("measurement_recheck_duplicate_case")
    for row in raw_packets:
        row_hash = row.get("recheck_packet_sha256")
        payload = {key: value for key, value in row.items() if key != "recheck_packet_sha256"}
        case_id = row.get("case_id")
        if (digest(payload) != row_hash or case_id not in packets or
                row.get("packet_sha256") != packets[case_id].get("packet_sha256") or
                row.get("question") != packets[case_id].get("question")):
            raise RuntimeError("measurement_recheck_packet_binding_invalid")
    outcomes = {}
    true_counts = {}
    for reviewer in ("a", "b"):
        rows = load(WORK / f"measurement-risk-recheck-{reviewer}-decisions.jsonl")
        if [r.get("case_id") for r in rows] != ids:
            raise RuntimeError(f"measurement_recheck_order_or_coverage_invalid:{reviewer}")
        by_id = {}
        true_counts[reviewer] = 0
        expected_ids = {row["case_id"]: row for row in raw_packets}
        for row in rows:
            case_id = row["case_id"]
            source = expected_ids[case_id]
            raw_source = next(line for line in packet_path.read_text(encoding="utf-8").splitlines()
                              if json.loads(line).get("case_id") == case_id)
            accepted_packet_hashes = {source["recheck_packet_sha256"], hashlib.sha256(raw_source.encode("utf-8")).hexdigest()}
            if (row.get("packet_sha256") != source["packet_sha256"] or
                    row.get("recheck_packet_sha256") not in accepted_packet_hashes or
                    row.get("measurement_applies") not in {True, False, "uncertain"} or
                    not isinstance(row.get("reason"), str) or not row["reason"].strip()):
                raise RuntimeError(f"measurement_recheck_decision_invalid:{reviewer}:{case_id}")
            question = source["question"].casefold()
            reason = row["reason"].casefold()
            quoted_phrases = re.findall(r"[“\"]([^”\"]+)[”\"]", row["reason"])
            quoted = any(phrase.strip().casefold() in question for phrase in quoted_phrases)
            if not quoted:
                raise RuntimeError(f"measurement_recheck_reason_not_question_bound:{reviewer}:{case_id}")
            by_id[case_id] = row["measurement_applies"]
            true_counts[reviewer] += row["measurement_applies"] is True
        outcomes[reviewer] = by_id
    overrides = {case_id for case_id in ids
                 if outcomes["a"][case_id] is True and outcomes["b"][case_id] is True}
    summary = {"candidate_cases": len(ids), "reviewer_a_true": true_counts["a"],
               "reviewer_b_true": true_counts["b"], "dual_true": len(overrides),
               "unresolved_or_disagreed": sum(outcomes["a"][c] != outcomes["b"][c] or
                                               outcomes["a"][c] == "uncertain" for c in ids),
               "applied_cases": sorted(overrides),
               "source_packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest()}
    return overrides, summary


def counts(rows: list[dict], risk_recheck_overrides: set[str] | None = None) -> dict:
    kinds = Counter()
    risks = Counter()
    risk_recheck_overrides = risk_recheck_overrides or set()
    for row in rows:
        p = row["packet"]
        kind = p["proposed_action"]["action"]
        if kind == "search":
            kind = "empty_evidence_search" if not p["visible_evidence"] else "nonempty_evidence_search"
        else:
            kind = "supported_answer"
        kinds[kind] += 1
        shared = normalized_risks(row["review_a"]["risk_labels"]) & normalized_risks(row["review_b"]["risk_labels"])
        if row["case_id"] in risk_recheck_overrides:
            shared.add("measurement")
        risks.update(shared)
    size = len(rows)
    return {"states": size, "questions": len({r["source_id"] for r in rows}),
            "state_kinds": dict(kinds), "risk_state_counts": dict(risks),
            "answer_fraction": kinds["supported_answer"] / size if size else 0,
            "nonempty_search_fraction": kinds["nonempty_evidence_search"] / size if size else 0,
            "empty_search_fraction": kinds["empty_evidence_search"] / size if size else 0}


def gate(summary: dict, split: str, frozen_internal: set[str]) -> list[str]:
    errors = []
    count = summary["states"]
    if split == "train" and not 300 <= count <= 400:
        errors.append("train_states_outside_300_400")
    if split == "internal" and count < 100:
        errors.append("internal_states_below_100")
    if split == "train" and summary["questions"] < 150:
        errors.append("train_questions_below_150")
    if split == "internal" and summary["questions"] != len(frozen_internal):
        errors.append("internal_frozen_question_coverage_incomplete")
    if summary["answer_fraction"] < .30:
        errors.append("answer_fraction_below_30pct")
    if summary["nonempty_search_fraction"] < .30:
        errors.append("nonempty_search_fraction_below_30pct")
    if summary["empty_search_fraction"] > .20:
        errors.append("empty_search_fraction_above_20pct")
    minimum = 20 if split == "train" else 5
    for risk in sorted(RISKS):
        if summary["risk_state_counts"].get(risk, 0) < minimum:
            errors.append(f"{risk}_risk_below_{minimum}")
    return errors


def select_cohort(rows: list[dict], split: str, frozen_internal: set[str],
                  risk_recheck_overrides: set[str] | None = None) -> list[dict] | None:
    """Choose the smallest deterministic approved cohort that meets every fixed gate."""
    minimum_size, maximum_size = (300, 400) if split == "train" else (100, len(rows))
    minimum_questions = 150 if split == "train" else len(frozen_internal)
    risk_recheck_overrides = risk_recheck_overrides or set()
    action_minimum = {}
    risks = sorted(RISKS)
    for size in range(minimum_size, maximum_size + 1):
        action_minimum = {"supported_answer": (size * 30 + 99) // 100,
                          "nonempty_evidence_search": (size * 30 + 99) // 100}
        risk_minimum = 20 if split == "train" else 5
        ordered = sorted(rows, key=lambda row: hashlib.sha256(
            (f"v114-cohort-v1:{split}:" + row["case_id"]).encode()).hexdigest())
        selected = []
        chosen = set()
        kind_counts = Counter()
        risk_counts = Counter()
        source_counts = Counter()

        def kind(row):
            action = row["packet"]["proposed_action"]["action"]
            if action == "answer":
                return "supported_answer"
            return "empty_evidence_search" if not row["packet"]["visible_evidence"] else "nonempty_evidence_search"

        def row_risks(row):
            labels = normalized_risks(row["review_a"]["risk_labels"]) & normalized_risks(row["review_b"]["risk_labels"])
            if row["case_id"] in risk_recheck_overrides:
                labels.add("measurement")
            return labels

        def deficits(row):
            labels = row_risks(row)
            action = kind(row)
            score = int(action in action_minimum and kind_counts[action] < action_minimum[action])
            score += sum(risk_counts[risk] < risk_minimum for risk in labels)
            return score

        required_sources = sorted(frozen_internal) if split == "internal" else sorted(
            {row["source_id"] for row in ordered},
            key=lambda value: hashlib.sha256(("v114-train-source-v1:" + value).encode()).hexdigest())[:minimum_questions]
        if split == "train" and len(required_sources) < minimum_questions:
            continue
        for source_id in required_sources:
            options = [row for row in ordered if row["source_id"] == source_id and row["case_id"] not in chosen]
            if not options:
                break
            row = max(options, key=lambda candidate: (deficits(candidate),
                                                       hashlib.sha256(("v114-source-pick:" + candidate["case_id"]).encode()).hexdigest()))
            selected.append(row)
            chosen.add(row["case_id"])
            kind_counts[kind(row)] += 1
            risk_counts.update(row_risks(row))
            source_counts[source_id] += 1
        if len(source_counts) < minimum_questions or len(selected) > size:
            continue
        while len(selected) < size:
            options = [row for row in ordered if row["case_id"] not in chosen]
            if not options:
                break
            row = max(options, key=lambda candidate: (deficits(candidate),
                                                       hashlib.sha256(("v114-fill:" + candidate["case_id"]).encode()).hexdigest()))
            selected.append(row)
            chosen.add(row["case_id"])
            kind_counts[kind(row)] += 1
            risk_counts.update(row_risks(row))
            source_counts[row["source_id"]] += 1
        summary = counts(selected, risk_recheck_overrides)
        if (summary["states"] == size and summary["questions"] >= minimum_questions
                and summary["answer_fraction"] >= .30 and summary["nonempty_search_fraction"] >= .30
                and summary["empty_search_fraction"] <= .20
                and all(summary["risk_state_counts"].get(risk, 0) >= risk_minimum for risk in risks)
                and (split != "internal" or {row["source_id"] for row in selected} == frozen_internal)):
            return selected
    return None


def main() -> None:
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        if hashlib.sha256((SOURCE / filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"source_hash_mismatch:{filename}")
    preflight = json.loads((WORK / "preflight.json").read_text(encoding="utf-8"))
    if hashlib.sha256((SOURCE / "manifest.json").read_bytes()).hexdigest() != preflight["source_manifest_sha256"]:
        raise RuntimeError("source_manifest_changed_after_preflight")
    training_freeze_path = WORK / "training-question-freeze-v2.json"
    base_training_freeze = json.loads((SOURCE / "training-question-freeze.json").read_text(encoding="utf-8"))
    if training_freeze_path.exists():
        frozen_train_record = json.loads(training_freeze_path.read_text(encoding="utf-8"))
        if (frozen_train_record.get("parent_freeze_sha256") != hashlib.sha256((SOURCE / "training-question-freeze.json").read_bytes()).hexdigest()
                or not set(base_training_freeze["ids"]).issubset(set(frozen_train_record.get("ids", [])))
                or frozen_train_record.get("question_count") != len(set(frozen_train_record.get("ids", [])))
                or len(frozen_train_record.get("question_hashes", [])) != len(set(frozen_train_record.get("ids", [])))):
            raise RuntimeError("training_question_freeze_v2_invalid")
    else:
        frozen_train_record = base_training_freeze
    frozen_train = set(frozen_train_record["ids"])
    reviewer_packets = {}
    for reviewer in ("a", "b"):
        path = WORK / f"reviewer-{reviewer}-packets.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != preflight["packet_files_sha256"][path.name]:
            raise RuntimeError(f"review_packet_file_changed:{reviewer}")
        rows = load(path)
        if len(rows) != 419 or len({r["case_id"] for r in rows}) != 419:
            raise RuntimeError(f"review_packet_coverage_invalid:{reviewer}")
        if any(digest({k: v for k, v in row.items() if k != "packet_sha256"}) != row["packet_sha256"] for row in rows):
            raise RuntimeError(f"review_packet_hash_invalid:{reviewer}")
        reviewer_packets[reviewer] = {r["case_id"]: r for r in rows}
    if reviewer_packets["a"] != reviewer_packets["b"]:
        raise RuntimeError("reviewers_received_different_cases")
    packets = reviewer_packets["a"]
    private = load(SOURCE / "private-key.jsonl")
    keys = {r["case_id"]: r for r in private}
    if set(keys) != set(packets):
        raise RuntimeError("private_key_case_mismatch")
    measurement_packets = {}
    measurement_keys = {}
    measurement_packet_path = WORK / "measurement-supplement-a-packets.jsonl"
    if measurement_packet_path.exists():
        measurement_rows = load(measurement_packet_path)
        for reviewer in ("a", "b"):
            path = WORK / f"measurement-supplement-{reviewer}-packets.jsonl"
            rows = load(path)
            if rows != measurement_rows or any(
                    digest({k: v for k, v in row.items() if k != "packet_sha256"}) != row.get("packet_sha256")
                    for row in rows):
                raise RuntimeError(f"measurement_supplement_packets_invalid:{reviewer}")
            measurement_packets[reviewer] = {row["case_id"]: row for row in rows}
        if measurement_packets["a"] != measurement_packets["b"]:
            raise RuntimeError("measurement_supplement_reviewer_packets_differ")
        measurement_private = load(WORK / "measurement-supplement-private.jsonl")
        measurement_keys = {row["case_id"]: row for row in measurement_private}
        if (set(measurement_keys) != set(measurement_packets["a"]) or
                any(row.get("split") != "train" or row.get("source_id") not in frozen_train
                    for row in measurement_private) or set(measurement_keys) & set(keys)):
            raise RuntimeError("measurement_supplement_private_mapping_invalid")
        packets.update(measurement_packets["a"])
        keys.update(measurement_keys)
    supplement_rows = {}
    for reviewer in ("a", "b"):
        rows = load(WORK / f"supplement-gap-{reviewer}-packets.jsonl")
        if len(rows) != 1:
            raise RuntimeError("expected_one_gap_supplement")
        row = rows[0]
        if digest({k: v for k, v in row.items() if k != "packet_sha256"}) != row["packet_sha256"]:
            raise RuntimeError("gap_supplement_hash_invalid")
        supplement_rows[reviewer] = row
    if supplement_rows["a"] != supplement_rows["b"]:
        raise RuntimeError("gap_reviewer_packets_differ")
    gap = supplement_rows["a"]
    gap_key = json.loads((WORK / "supplement-gap-private.json").read_text(encoding="utf-8"))
    if gap_key["case_id"] != gap["case_id"] or gap_key["case_id"] in keys or gap_key["split"] != "internal":
        raise RuntimeError("gap_private_mapping_invalid")
    packets = {**packets, gap["case_id"]: gap}
    keys[gap["case_id"]] = gap_key
    replacement_packets = {}
    replacement_keys = {}
    replacement_ids: list[str] = []
    replacement_freeze_path = WORK / "internal-freeze-v2.json"
    if replacement_freeze_path.exists():
        replacement_freeze = json.loads(replacement_freeze_path.read_text(encoding="utf-8"))
        old_freeze = json.loads((SOURCE / "internal-freeze.json").read_text(encoding="utf-8"))
        replacement_info = json.loads((WORK / "replacement-question.json").read_text(encoding="utf-8"))
        if (len(replacement_freeze["ids"]) != 50 or
                replacement_freeze["replaced_source_id"] not in old_freeze["ids"] or
                replacement_freeze["replaced_source_id"] in replacement_freeze["ids"] or
                replacement_freeze["replacement_source_id"] not in replacement_freeze["ids"] or
                set(replacement_freeze["ids"]) != (set(old_freeze["ids"]) - {replacement_freeze["replaced_source_id"]}) | {replacement_freeze["replacement_source_id"]} or
                replacement_info["source_id"] != replacement_freeze["replacement_source_id"]):
            raise RuntimeError("replacement_freeze_invalid")
        if hashlib.sha256(norm(replacement_info["question"]).encode()).hexdigest() != replacement_info["normalized_question_sha256"]:
            raise RuntimeError("replacement_question_hash_invalid")
        for reviewer in ("a", "b"):
            rows = load(WORK / f"replacement-{reviewer}-packets.jsonl")
            if hashlib.sha256((b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                                     separators=(",", ":")).encode() + b"\n" for row in rows))).hexdigest() != replacement_info["packet_sha256"]:
                raise RuntimeError(f"replacement_packet_file_hash_invalid:{reviewer}")
            if any(digest({k: v for k, v in row.items() if k != "packet_sha256"}) != row["packet_sha256"] for row in rows):
                raise RuntimeError(f"replacement_packet_hash_invalid:{reviewer}")
            replacement_packets[reviewer] = {row["case_id"]: row for row in rows}
        if replacement_packets["a"] != replacement_packets["b"]:
            raise RuntimeError("replacement_reviewer_packets_differ")
        replacement_ids = list(replacement_packets["a"])
        replacement_private = load(WORK / "replacement-private.jsonl")
        replacement_keys = {row["case_id"]: row for row in replacement_private}
        if set(replacement_keys) != set(replacement_ids) or any(
                row["split"] != "internal" or row["source_id"] != replacement_info["source_id"]
                for row in replacement_private):
            raise RuntimeError("replacement_private_mapping_invalid")
        if set(replacement_ids) & set(packets):
            raise RuntimeError("replacement_case_id_collision")
        packets.update(replacement_packets["a"])
        keys.update(replacement_keys)
    internal_v3_freeze_path = WORK / "internal-freeze-v3.json"
    internal_v3_ids: list[str] = []
    internal_v3_packets = {}
    internal_v3_freeze = None
    if internal_v3_freeze_path.exists():
        internal_v3_freeze = json.loads(internal_v3_freeze_path.read_text(encoding="utf-8"))
        if (internal_v3_freeze.get("version") != "v114-internal-freeze-v3"
                or internal_v3_freeze.get("parent_freeze_sha256") != hashlib.sha256(replacement_freeze_path.read_bytes()).hexdigest()
                or len(internal_v3_freeze.get("ids", [])) != 50
                or len(set(internal_v3_freeze.get("ids", []))) != 50
                or len(internal_v3_freeze.get("question_hashes", [])) != 50):
            raise RuntimeError("internal_freeze_v3_invalid")
        expected_ids = ((set(replacement_freeze["ids"]) - set(internal_v3_freeze.get("replaced_source_ids", [])))
                        | set(internal_v3_freeze.get("replacement_source_ids", [])))
        if (len(internal_v3_freeze.get("replaced_source_ids", [])) != 3
                or len(internal_v3_freeze.get("replacement_source_ids", [])) != 3
                or set(internal_v3_freeze["ids"]) != expected_ids
                or set(internal_v3_freeze.get("replacement_source_ids", [])) & frozen_train):
            raise RuntimeError("internal_freeze_v3_membership_invalid")
        v3_manifest = json.loads((WORK / "internal-replacements-v3-manifest.json").read_text(encoding="utf-8"))
        if (v3_manifest.get("freeze_sha256") != hashlib.sha256(internal_v3_freeze_path.read_bytes()).hexdigest()
                or v3_manifest.get("source_ids") != internal_v3_freeze["replacement_source_ids"]
                or v3_manifest.get("audit_ids") != internal_v3_freeze.get("replacement_audit_ids")):
            raise RuntimeError("internal_replacements_v3_manifest_invalid")
        for reviewer in ("a", "b"):
            path = WORK / f"internal-replacements-v3-{reviewer}-packets.jsonl"
            rows = load(path)
            raw = path.read_bytes()
            if (hashlib.sha256(raw).hexdigest() != v3_manifest.get("packet_sha256")
                    or any(digest({k: v for k, v in row.items() if k != "packet_sha256"}) != row.get("packet_sha256")
                           for row in rows)):
                raise RuntimeError(f"internal_replacements_v3_packet_invalid:{reviewer}")
            internal_v3_packets[reviewer] = {row["case_id"]: row for row in rows}
        if internal_v3_packets["a"] != internal_v3_packets["b"]:
            raise RuntimeError("internal_replacements_v3_reviewer_packet_mismatch")
        internal_v3_ids = list(internal_v3_packets["a"])
        internal_v3_private = load(WORK / "internal-replacements-v3-private.jsonl")
        internal_v3_keys = {row["case_id"]: row for row in internal_v3_private}
        if (set(internal_v3_keys) != set(internal_v3_ids)
                or any(row.get("split") != "internal"
                       or row.get("source_id") not in set(internal_v3_freeze["replacement_source_ids"])
                       for row in internal_v3_private)
                or set(internal_v3_ids) & set(packets)):
            raise RuntimeError("internal_replacements_v3_private_mapping_invalid")
        packets.update(internal_v3_packets["a"])
        keys.update(internal_v3_keys)
    reserve_inventory = json.loads((WORK / "reserve-inventory.json").read_text(encoding="utf-8"))
    for filename, expected in reserve_inventory["files_sha256"].items():
        if hashlib.sha256((WORK / filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"reserve_inventory_changed:{filename}")
    reserve_keys = {r["case_id"]: r for r in load(WORK / "reserve-private.jsonl")}
    dispatched_waves = []
    staged_waves = []
    for manifest_path in sorted(WORK.glob("reserve-wave-*-manifest.json")):
        wave = json.loads(manifest_path.read_text(encoding="utf-8"))
        if hashlib.sha256((WORK / "reserve-inventory.json").read_bytes()).hexdigest() != wave["inventory_sha256"]:
            raise RuntimeError("reserve_wave_inventory_changed")
        wave_rows = {}
        for reviewer in ("a", "b"):
            path = WORK / f"reserve-wave-{wave['wave']}-{reviewer}-packets.jsonl"
            if hashlib.sha256(path.read_bytes()).hexdigest() != wave["packet_sha256"]:
                raise RuntimeError("reserve_wave_packet_changed")
            wave_rows[reviewer] = load(path)
        if wave_rows["a"] != wave_rows["b"] or [r["case_id"] for r in wave_rows["a"]] != wave["case_ids"]:
            raise RuntimeError("reserve_wave_reviewer_mismatch")
        if not any((WORK / f"reserve-wave-{wave['wave']}-{reviewer}-decisions.jsonl").exists() for reviewer in ("a", "b")):
            staged_waves.append(wave["wave"])
            continue
        for row in wave_rows["a"]:
            case_id = row["case_id"]
            if digest({k: v for k, v in row.items() if k != "packet_sha256"}) != row["packet_sha256"]:
                raise RuntimeError("reserve_packet_hash_invalid")
            if case_id in packets or case_id not in reserve_keys:
                raise RuntimeError("reserve_duplicate_or_unknown_case")
            packets[case_id] = row
            private_row = dict(reserve_keys[case_id])
            if replacement_freeze_path.exists() and private_row["source_id"] == replacement_freeze["replaced_source_id"]:
                private_row["split"] = "retired_internal"
            if internal_v3_freeze and private_row["source_id"] in set(internal_v3_freeze["replaced_source_ids"]):
                private_row["split"] = "retired_internal"
            keys[case_id] = private_row
        dispatched_waves.append(wave)
    frozen_path = internal_v3_freeze_path if internal_v3_freeze else (
        replacement_freeze_path if replacement_freeze_path.exists() else SOURCE / "internal-freeze.json")
    frozen_internal_record = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_internal = set(frozen_internal_record["ids"])
    for private_row in keys.values():
        if private_row.get("split") == "internal" and private_row.get("source_id") not in frozen_internal:
            private_row["split"] = "retired_internal"
    freeze_isolation_errors = []
    if frozen_internal & frozen_train:
        freeze_isolation_errors.append("frozen_source_id_overlap")
    if set(frozen_internal_record.get("question_hashes", [])) & set(frozen_train_record.get("question_hashes", [])):
        freeze_isolation_errors.append("frozen_normalized_question_overlap")
    quality = json.loads((WORK / "review-quality-quarantine.json").read_text(encoding="utf-8"))
    base_a, errors_a = load_decisions("a", reviewer_packets["a"], expected_order=list(reviewer_packets["a"]))
    base_b, errors_b = load_decisions("b", reviewer_packets["b"], WORK / quality["reviewer_b_authoritative_file"],
                                      expected_order=list(reviewer_packets["b"]))
    append_only_corrections, errors_a = apply_battlefield_rereview(base_a, errors_a, reviewer_packets["a"])
    for case_id in quality["reviewer_a_invalid_case_ids"]:
        if case_id in base_a:
            base_a[case_id]["_invalid_errors"].append("reviewer_self_reported_misaligned")
            errors_a.append({"case_id": case_id, "reviewer": "a", "error": ["reviewer_self_reported_misaligned"]})
    gap_a, gap_errors_a = load_decisions("a", {gap["case_id"]: gap}, WORK / "supplement-gap-a-decisions.jsonl", [gap["case_id"]])
    gap_b, gap_errors_b = load_decisions("b", {gap["case_id"]: gap}, WORK / "supplement-gap-b-decisions.jsonl", [gap["case_id"]])
    decisions_a, decisions_b = {**base_a, **gap_a}, {**base_b, **gap_b}
    errors_a += gap_errors_a
    errors_b += gap_errors_b
    measurement_supplement_ids = list(measurement_packets.get("a", {}))
    if measurement_supplement_ids:
        normalized_measurement_a = normalize_luna_reviews(
            "a", "measurement-supplement", measurement_packets["a"], measurement_supplement_ids)
        normalized_measurement_b = normalize_luna_reviews(
            "b", "measurement-supplement", measurement_packets["b"], measurement_supplement_ids)
        supplemental_a, supplemental_a_errors = load_decisions(
            "a", measurement_packets["a"], normalized_measurement_a,
            measurement_supplement_ids)
        supplemental_b, supplemental_b_errors = load_decisions(
            "b", measurement_packets["b"], normalized_measurement_b,
            measurement_supplement_ids)
        decisions_a.update(supplemental_a)
        decisions_b.update(supplemental_b)
        errors_a.extend(supplemental_a_errors)
        errors_b.extend(supplemental_b_errors)
    if replacement_ids:
        replacement_a, replacement_errors_a = load_decisions(
            "a", replacement_packets["a"], WORK / "replacement-a-decisions.jsonl", replacement_ids)
        replacement_b, replacement_errors_b = load_decisions(
            "b", replacement_packets["b"], WORK / "replacement-b-decisions.jsonl", replacement_ids)
        decisions_a.update(replacement_a)
        decisions_b.update(replacement_b)
        errors_a.extend(replacement_errors_a)
        errors_b.extend(replacement_errors_b)
    if internal_v3_ids:
        normalized_v3_a = normalize_luna_reviews(
            "a", "internal-replacements-v3", internal_v3_packets["a"], internal_v3_ids)
        normalized_v3_b = normalize_luna_reviews(
            "b", "internal-replacements-v3", internal_v3_packets["b"], internal_v3_ids)
        v3_a, v3_errors_a = load_decisions(
            "a", internal_v3_packets["a"], normalized_v3_a, internal_v3_ids)
        v3_b, v3_errors_b = load_decisions(
            "b", internal_v3_packets["b"], normalized_v3_b, internal_v3_ids)
        decisions_a.update(v3_a)
        decisions_b.update(v3_b)
        errors_a.extend(v3_errors_a)
        errors_b.extend(v3_errors_b)
    for wave in dispatched_waves:
        wave_packets = {case_id: packets[case_id] for case_id in wave["case_ids"]}
        for reviewer, decisions, errors in (("a", decisions_a, errors_a), ("b", decisions_b, errors_b)):
            fresh, bad = load_decisions(reviewer, wave_packets,
                                        WORK / f"reserve-wave-{wave['wave']}-{reviewer}-decisions.jsonl",
                                        expected_order=wave["case_ids"])
            decisions.update(fresh)
            errors.extend(bad)
    risk_recheck_overrides, risk_recheck_summary = load_measurement_risk_recheck(packets)
    reviewer_a_ceiling = None
    reviewer_a_ceiling_detail = None
    if len(base_a) == 419:
        reviewer_a_ceiling = {split: sum(r["approved"] and not r["_invalid_errors"] and
                                        keys[case_id]["split"] == split
                                        for case_id, r in base_a.items()) for split in ("train", "internal")}
        reviewer_a_ceiling_detail = {}
        for split in ("train", "internal"):
            rows = [(case_id, review) for case_id, review in base_a.items()
                    if review["approved"] and not review["_invalid_errors"] and keys[case_id]["split"] == split]
            kinds = Counter()
            risks = Counter()
            for case_id, review in rows:
                packet = reviewer_packets["a"][case_id]
                action = packet["proposed_action"]["action"]
                kind = "supported_answer" if action == "answer" else (
                    "nonempty_evidence_search" if packet["visible_evidence"] else "empty_evidence_search")
                kinds[kind] += 1
                risks.update(normalized_risks(review["risk_labels"]))
            reviewer_a_ceiling_detail[split] = {
                "states": len(rows), "questions": len({keys[c]["source_id"] for c, _ in rows}),
                "state_kinds": dict(kinds), "risk_state_counts": dict(risks),
            }
    accepted = []
    disputed = []
    rejected = []
    invalid = []
    pending = []
    retired = []
    for case_id, packet in packets.items():
        a, b = decisions_a.get(case_id), decisions_b.get(case_id)
        if a is None or b is None:
            pending.append(case_id)
        elif a["_invalid_errors"] or b["_invalid_errors"]:
            invalid.append(case_id)
        elif a["approved"] and b["approved"]:
            if keys[case_id]["split"] == "retired_internal":
                retired.append(case_id)
            else:
                accepted.append({"case_id": case_id, "source_id": keys[case_id]["source_id"],
                                 "split": keys[case_id]["split"], "packet": packet,
                                 "review_a": a, "review_b": b})
        elif a["approved"] != b["approved"]:
            disputed.append(case_id)
        else:
            rejected.append(case_id)
    def review_subset_status(case_ids: list[str]) -> dict:
        ids = set(case_ids)
        return {
            "candidate_states": len(ids),
            "reviewer_a_complete": sum(case_id in decisions_a for case_id in ids),
            "reviewer_b_complete": sum(case_id in decisions_b for case_id in ids),
            "dual_approved_valid": sum(row["case_id"] in ids for row in accepted),
            "rejected_by_both": sum(case_id in ids for case_id in rejected),
            "disputed": sum(case_id in ids for case_id in disputed),
            "invalid_quarantined": sum(case_id in ids for case_id in invalid),
            "pending": sum(case_id in ids for case_id in pending),
        }
    measurement_review_status = review_subset_status(list(measurement_packets.get("a", {})))
    internal_v3_review_status = review_subset_status(internal_v3_ids)
    train_pool = [r for r in accepted if r["split"] == "train"]
    internal_pool = [r for r in accepted if r["split"] == "internal"]
    accepted_train_cases = {row["case_id"] for row in train_pool}
    risk_recheck_overrides &= accepted_train_cases
    risk_recheck_summary["applied_cases"] = sorted(risk_recheck_overrides)
    risk_recheck_summary["applied_count"] = len(risk_recheck_overrides)
    train_cohort = select_cohort(train_pool, "train", frozen_internal, risk_recheck_overrides)
    internal_cohort = select_cohort(internal_pool, "internal", frozen_internal, risk_recheck_overrides)
    train = train_cohort if train_cohort is not None else train_pool
    internal = internal_cohort if internal_cohort is not None else internal_pool
    isolation_errors = []
    train_ids, internal_ids = {r["source_id"] for r in train}, {r["source_id"] for r in internal}
    if train_ids & internal_ids or not train_ids.issubset(frozen_train) or not internal_ids.issubset(frozen_internal):
        isolation_errors.append("source_id_overlap_or_outside_freeze")
    qtrain = {norm(r["packet"]["question"]) for r in train}
    qinternal = {norm(r["packet"]["question"]) for r in internal}
    if qtrain & qinternal:
        isolation_errors.append("normalized_question_overlap")
    all_train_q = {norm(packet["question"]) for case_id, packet in packets.items() if keys[case_id]["split"] == "train"}
    all_internal_q = {norm(packet["question"]) for case_id, packet in packets.items()
                      if keys[case_id]["split"] == "internal" and keys[case_id]["source_id"] in frozen_internal}
    if all_train_q & all_internal_q:
        isolation_errors.append("candidate_normalized_question_overlap")
    train_summary, internal_summary = counts(train, risk_recheck_overrides), counts(internal, risk_recheck_overrides)
    gate_errors = gate(train_summary, "train", frozen_internal) + gate(internal_summary, "internal", frozen_internal)
    if train_cohort is None:
        gate_errors.append("no_train_cohort_meets_all_gates")
    if internal_cohort is None:
        gate_errors.append("no_internal_cohort_meets_all_gates")
    gate_errors.extend(freeze_isolation_errors)
    gap_decisions = {"reviewer_a": decisions_a.get(gap["case_id"]), "reviewer_b": decisions_b.get(gap["case_id"])}
    gap_both_rejected = all(r is not None and not r["approved"] for r in gap_decisions.values())
    if gap_both_rejected and gap_key["source_id"] in frozen_internal and gap_key["source_id"] not in internal_ids:
        gate_errors.append("frozen_gap_state_dual_rejected")
    if reviewer_a_ceiling and not dispatched_waves:
        if reviewer_a_ceiling["train"] < 300:
            gate_errors.append("original_candidates_train_upper_bound_below_300")
        if reviewer_a_ceiling["internal"] < 100:
            gate_errors.append("original_candidates_internal_upper_bound_below_100")
    # A reserve wave may remain unopened in part once a fully dual-reviewed
    # minimum cohort already passes every gate.  Unselected rows stay pending
    # and are reported, but do not block an otherwise complete data package.
    review_stopped_at_gate = bool(pending and train_cohort is not None and internal_cohort is not None)
    if pending and not review_stopped_at_gate:
        gate_errors.append("dual_review_incomplete")
    if any(isinstance(item["error"], str) and item["error"] in {"foreign_case", "decision_order_mismatch"}
           for item in errors_a + errors_b):
        gate_errors.append("review_integrity_error")
    gate_errors.extend(isolation_errors)
    report = {
        "version": "v114", "review_scope": len(packets), "original_candidate_count": 419,
        "gap_supplement_count": 1, "active_reserve_waves": [w["wave"] for w in dispatched_waves],
        "internal_freeze": {"version": "v3" if internal_v3_freeze else ("v2" if replacement_freeze_path.exists() else "v1"),
                            "replaced_source_id": (replacement_freeze.get("replaced_source_id") if replacement_freeze_path.exists() else None),
                            "replacement_source_id": (replacement_freeze.get("replacement_source_id") if replacement_freeze_path.exists() else None),
                            "replacement_candidate_states": len(replacement_ids),
                            "v3_replaced_source_ids": (internal_v3_freeze.get("replaced_source_ids", []) if internal_v3_freeze else []),
                            "v3_replacement_source_ids": (internal_v3_freeze.get("replacement_source_ids", []) if internal_v3_freeze else []),
                            "v3_replacement_candidate_states": len(internal_v3_ids)},
        "training_question_freeze": {"version": frozen_train_record.get("version", "v114"),
                                     "question_count": len(frozen_train),
                                     "added_source_ids": frozen_train_record.get("added_source_ids", [])},
        "measurement_supplement_candidate_count": len(measurement_supplement_ids),
        "measurement_supplement_review": measurement_review_status,
        "internal_v3_replacement_review": internal_v3_review_status,
        "staged_reserve_waves": staged_waves,
        "original_419_coverage": {"reviewer_a": len(base_a), "reviewer_b": len(base_b),
                                   "both": len(set(base_a) & set(base_b))},
        "original_419_dual_approval_upper_bound_from_reviewer_a": reviewer_a_ceiling,
        "original_419_reviewer_a_upper_bound_detail": reviewer_a_ceiling_detail,
        "original_plus_gap_internal_candidates": {"states": sum(keys[c]["split"] == "internal" for c in reviewer_packets["a"]) + 1,
                                                   "questions": len({keys[c]["source_id"] for c in reviewer_packets["a"] if keys[c]["split"] == "internal"} | {gap_key["source_id"]})},
        "coverage": {"reviewer_a": len(decisions_a), "reviewer_b": len(decisions_b), "both": len(packets) - len(pending),
                     "pending": len(pending), "dual_approved": len(accepted), "rejected_by_both": len(rejected),
                     "disputed": len(disputed), "invalid_quarantined": len(invalid),
                     "retired_ambiguous_question_dual_approved": len(retired)},
        "approved_pool": {"train_states": len(train_pool), "internal_states": len(internal_pool)},
        "selected_sft_cohort": {"train_states": len(train_cohort) if train_cohort is not None else 0,
                                "internal_states": len(internal_cohort) if internal_cohort is not None else 0,
                                "deterministic_minimum_size": True},
        "review_stopped_at_gate": review_stopped_at_gate,
        "train": train_summary, "internal": internal_summary,
        "missing_internal_ids": sorted(frozen_internal - internal_ids),
        "gap_state": {"case_id": gap["case_id"], "source_id": gap_key["source_id"],
                      "dual_rejected": gap_both_rejected,
                      "approved_internal_coverage": gap_key["source_id"] in internal_ids},
        "candidate_question_isolation": {"train_unique_normalized": len(all_train_q),
                                         "internal_unique_normalized": len(all_internal_q),
                                         "overlap": len(all_train_q & all_internal_q)},
        "invalid_review_records": errors_a + errors_b,
        "append_only_corrections": append_only_corrections,
        "risk_category_recheck": risk_recheck_summary,
        "review_quality_quarantine": quality,
        "risk_label_normalization": {"entity": "relation",
                                     "reason": "Entity-to-role identity decisions are counted in the frozen relation risk category; raw reviewer labels remain unchanged in the audit log."},
        "gate_errors": sorted(set(gate_errors)), "passed": not gate_errors,
        "training_permitted": not gate_errors, "independent100_evaluated": False,
        "final_test_accessed": False,
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, values in (("accepted", accepted), ("disputed", disputed), ("rejected", rejected),
                         ("invalid", invalid), ("pending", pending), ("retired", retired)):
        (WORK / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values), encoding="utf-8")
    if report["training_permitted"]:
        for filename, cohort in (("sft-train.jsonl", train_cohort), ("sft-internal.jsonl", internal_cohort)):
            records = []
            for row in cohort:
                messages = list(row["packet"]["visible_messages"])
                messages.append({"role": "assistant", "content": json.dumps(
                    row["packet"]["proposed_action"], ensure_ascii=False, separators=(",", ":"))})
                records.append({"messages": messages, "target_assistant_index": len(messages) - 1,
                                "thinking_enabled": False})
            (WORK / filename).write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    print(json.dumps({"coverage": report["coverage"], "train": train_summary, "internal": internal_summary,
                      "gate_errors": report["gate_errors"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
