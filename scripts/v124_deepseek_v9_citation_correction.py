"""One citation-only correction for v9 batch 01; decisions are immutable."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request

from scripts.v123_cpu_core import sha, write
from scripts.v124_deepseek_blind_review import env_values
from scripts.v124_forensic import OUT


BASE = OUT / "pending-dual-review-v9"
PACKET_PATH = BASE / "v9-batch-01-packet-private.json"
REVIEW_PATH = BASE / "deepseek-responses/v9-batch-01/validated-review-private.json"
JOURNAL = BASE / "deepseek-journals/v9-batch-01-citation-correction-private.jsonl"
ATTEMPTS = BASE / "deepseek-responses/v9-batch-01-citation-correction"
ENV_FILE = OUT.parent.parent.parent / ".env"
MODEL = "deepseek-flash"


def append(row: dict) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


def main() -> None:
    packet_obj = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    payload_hash = packet_obj.pop("packet_sha256")
    original = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    if original["packet_sha256"] != payload_hash:
        raise RuntimeError("original_vote_packet_mismatch")
    if JOURNAL.exists() and JOURNAL.stat().st_size:
        raise RuntimeError("citation_correction_already_attempted_stop")

    old = {r["state_id"]: r for r in original["reviews"]}
    states = {s["state_id"]: s for g in packet_obj["groups"] for s in g["states"]}
    if set(old) != set(states):
        raise RuntimeError("original_vote_state_coverage_mismatch")
    schema_fields = ("decision", "issues", "rationale", "query_grounded",
                     "relevant_new_evidence", "answer_supported")
    out_schema = {"reviews": [{
        "state_id": "exact existing state_id",
        "decision": "copy exactly; do not revise",
        "issues": "copy exactly; do not revise",
        "rationale": "copy exactly; do not revise",
        "citations": [{"stage": "exact allowed stage", "document_id": "exact ID", "quote": "verbatim"}],
        "query_grounded": "copy exactly; do not revise",
        "relevant_new_evidence": "copy exactly; do not revise",
        "answer_supported": "copy exactly; do not revise",
    }]}
    system = (
        "Perform a citation-only technical correction. This is not a second semantic review. "
        "For every state copy decision, issues, rationale, and boolean judgments exactly from the "
        "original vote. Do not change wording or conclusions. Add precise citations that substantiate "
        "the already-written rationale. For an approved search, cite both the action-time grounding "
        "(visible user question and/or pre-action evidence) and an exact post-action result that supports "
        "the claim of relevant new evidence. If the packet contains no result that supports the rationale, "
        "do not invent one; keep the original wording and return only citations that truly exist. "
        "Allowed citation stages are exactly visible_messages, pre_action_evidence, "
        "post_action_retrieval_results. For a visible user message document_id must be user; for evidence "
        "copy the exact D-number. Every quote must be a verbatim substring of that exact stage/document. "
        "Return a JSON object matching the supplied schema. No other changes are allowed."
    )
    user = {"packet": packet_obj, "original_review": original["reviews"],
            "immutable_fields": list(schema_fields), "required_output_schema": out_schema}
    body_obj = {"model": MODEL, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, separators=(",", ":"))}],
        "temperature": 0, "max_tokens": 12000,
        "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"}}
    body = json.dumps(body_obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request_hash = hashlib.sha256(body).hexdigest()
    ATTEMPTS.mkdir(parents=True, exist_ok=True)
    append({"packet_sha256": payload_hash, "model": MODEL, "request_sha256": request_hash,
            "status": "citation_only_correction_sending", "started_unix": int(time.time()),
            "decisions_immutable": True, "correction_limit": 1})
    base = env_values(ENV_FILE)
    key = base.get("DEEPSEEK_API_KEY", "").strip()
    api_base = base.get("DEEPSEEK_API_BASE", "https://api.deepseek.com").strip().rstrip("/")
    if not key:
        raise RuntimeError("deepseek_api_key_missing_from_explicit_env")
    url = api_base if api_base.endswith("/chat/completions") else api_base + "/chat/completions"
    request = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            raw, http_status = response.read(), response.status
    except urllib.error.HTTPError as error:
        append({"packet_sha256": payload_hash, "status": "http_error_response_received",
                "http_status": error.code, "finished_unix": int(time.time())})
        raise RuntimeError("http_error_response_received:" + str(error.code)) from None
    except Exception:
        append({"packet_sha256": payload_hash, "status": "outcome_unknown_no_retry",
                "finished_unix": int(time.time())})
        raise RuntimeError("outcome_unknown_no_retry") from None
    raw_path = ATTEMPTS / "citation-correction-raw.json"
    raw_path.write_bytes(raw)
    append({"packet_sha256": payload_hash, "status": "response_received",
            "http_status": http_status, "response_sha256": sha(raw_path),
            "finished_unix": int(time.time())})
    try:
        outer = json.loads(raw)
        corrected = json.loads(outer["choices"][0]["message"]["content"])
        rows = corrected.get("reviews")
        if not isinstance(rows, list) or {r.get("state_id") for r in rows} != set(old):
            raise ValueError("correction_state_coverage_invalid")
        corrected_by_id = {r["state_id"]: r for r in rows}
        for state_id, row in corrected_by_id.items():
            prev = old[state_id]
            if any(row.get(key) != prev.get(key) for key in schema_fields):
                raise ValueError("semantic_fields_changed:" + state_id)
            state = states[state_id]
            docs = {}
            for d in state["pre_action_evidence"]:
                docs[("pre_action_evidence", d["document_id"])] = d["text"]
            for d in state["post_action_retrieval_results"] or []:
                docs[("post_action_retrieval_results", d["document_id"])] = d["text"]
            for msg in state["visible_messages"]:
                if msg["role"] == "user":
                    docs[("visible_messages", "user")] = msg["content"]
            if not isinstance(row.get("citations"), list):
                raise ValueError("correction_citations_invalid:" + state_id)
            for cite in row["citations"]:
                key = (cite.get("stage"), cite.get("document_id"))
                quote = cite.get("quote")
                if key not in docs or not isinstance(quote, str) or not quote or quote not in docs[key]:
                    raise ValueError("correction_citation_invalid:" + state_id)
        result = {"packet_sha256": payload_hash, "original_review_sha256": sha(REVIEW_PATH),
                  "model": MODEL, "reviews": rows,
                  "correction_type": "citations_only",
                  "decisions_and_rationales_unchanged": True,
                  "training_allowed": False}
        out_path = ATTEMPTS / "citation-correction-validated-private.json"
        write(out_path, result)
        append({"packet_sha256": payload_hash, "status": "citation_correction_saved",
                "correction_sha256": sha(out_path), "finished_unix": int(time.time()),
                "training_allowed": False})
        print(json.dumps({"status": "citation_correction_saved", "states": len(rows),
                          "correction_sha256": sha(out_path), "training_allowed": False}, ensure_ascii=False))
    except Exception as exc:
        append({"packet_sha256": payload_hash, "status": "invalid_citation_correction",
                "reason": type(exc).__name__ + ":" + str(exc)[:160],
                "finished_unix": int(time.time())})
        raise RuntimeError("invalid_citation_correction_stop_without_retry") from None


if __name__ == "__main__":
    main()
