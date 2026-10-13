"""Explicit, resumable DeepSeek transport for a single frozen v123 CPU batch."""
from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from urllib.parse import urlparse

from scripts.v123_cpu_core import ROOT, OUT, digest, read, readl, sha
from scripts.v123_review import REVIEW, append_vote, all_votes, packets_and_source


def audit_batch(batch_name: str) -> dict:
    import httpx
    from scripts.v123_api_protocol import SYSTEM, call

    if not batch_name.startswith("work-api-") or not batch_name.endswith(".jsonl") or "/" in batch_name or "\\" in batch_name:
        raise ValueError("not_an_api_batch_name")
    batch = REVIEW / batch_name
    manifest = read(batch.with_suffix(".manifest.json"))
    if manifest["batch"] != batch_name or manifest["reviewer"] != "api" or manifest["packet_file_sha256"] != sha(batch):
        raise ValueError("audit_batch_binding_changed")
    payloads = readl(batch)
    if len(payloads) != manifest["cases"] or len(payloads) > 20:
        raise ValueError("audit_batch_size_invalid")
    _, packets, _ = packets_and_source()
    by_case = {p["case_id"]: p for p in packets}
    current = all_votes("api", packets)
    config = {key: os.environ.get(key, "") for key in
              ("DEEPSEEK_API_BASE", "DEEPSEEK_API_KEY", "MODEL")}
    endpoint = config.get("DEEPSEEK_API_BASE", "").rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or parsed.hostname != "api.deepseek.com" or parsed.path not in {"", "/"}:
        raise ValueError("authorized_deepseek_host_required")
    model = config.get("MODEL", "")
    if not config.get("DEEPSEEK_API_KEY") or not model:
        raise ValueError("review_api_configuration_missing")
    journal = REVIEW / "api-request-journal.jsonl"
    events = readl(journal) if journal.exists() else []
    started = {r["case_id"] for r in events if r["event"] == "started"}
    finished = {r["case_id"] for r in events if r["event"] == "finished"}
    unsettled = started - finished - set(current)
    if unsettled:
        raise ValueError("unsettled_api_request_requires_manual_resolution")
    completed = 0
    with httpx.Client(timeout=180, trust_env=False) as client:
        for public in payloads:
            case = public["case_id"]
            private = by_case.get(case)
            if private is None or public != {key: private[key] for key in public}:
                raise ValueError("public_packet_changed_or_unknown")
            if case in current:
                continue
            request = {"event": "started", "case_id": case, "packet_sha256": digest(private),
                       "batch_sha256": manifest["packet_file_sha256"], "model": model,
                       "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest()}
            with journal.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(request, ensure_ascii=False) + "\n")
                stream.flush()
            errors = []
            decision = None
            for attempt in range(2):
                try:
                    decision = call(client, config["DEEPSEEK_API_KEY"], endpoint + "/chat/completions",
                                    model, public, errors[-1]["message"] if attempt else None)
                    break
                except (ValueError, KeyError) as error:
                    errors.append({"kind": type(error).__name__, "message": str(error)[:180]})
                # Network errors propagate.  The unfinished journal prevents an uncertain retry.
            append_vote("api", case, decision, digest(private), model, errors)
            with journal.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"event": "finished", "case_id": case,
                                         "valid_response": decision is not None}, ensure_ascii=False) + "\n")
                stream.flush()
            current[case] = True
            completed += 1
    return {"version": "v123-api-batch-v1", "batch": batch_name,
            "completed": completed, "total": len(payloads), "training_allowed": False}
