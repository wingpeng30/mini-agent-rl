"""Forward-only DeepSeek reviewer for v129 packets; never resumes a sent batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from scripts.v130_review_protocol import SYSTEM, contract_identity, valid_vote
from scripts.v123_cpu_core import digest

ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / "data/evidence-sft-v124-forensic/goal-v128"
AUDIT_ROOT = GOAL / "new-internal-audit"
ENV_FILE = ROOT / ".env"
WORKSPACE_ENV_FILE = ROOT.parent / ".env"
MODEL = "deepseek-flash"
REQUEST_MAX_TOKENS = 8192
REQUEST_THINKING = {"type": "disabled"}
MAX_TECHNICAL_CORRECTIONS = 1
FORBIDDEN_PATH_PARTS = ("final-test", "final_test", "finaltest", "independent100",
                        "independent-100", "independent_100", "holdout100", "holdout-100")
ARCHIVE_SUFFIXES = {".zip", ".7z", ".rar", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".zst"}


def _append(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _credentials() -> tuple[str, str]:
    """Resolve the API credential without persisting or emitting its value."""
    config = {}
    for path in (WORKSPACE_ENV_FILE, ENV_FILE):
        if path.is_file():
            config.update(_read_env(path))
    key = os.environ.get("DEEPSEEK_API_KEY", config.get("DEEPSEEK_API_KEY", "")).strip()
    base = os.environ.get("DEEPSEEK_API_BASE", config.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    return key, base.strip().rstrip("/")


def _payload_hash(packets: list[dict]) -> str:
    return digest({"protocol": contract_identity(), "packets": packets})


def checked_packet_path(value: str) -> Path:
    """Reject paths lexically before resolving or reading any candidate file."""
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts or ":" in value or raw.suffix.casefold() in ARCHIVE_SUFFIXES:
        raise ValueError("packet_path_rejected_before_read")
    if any(any(token in part.casefold() for token in FORBIDDEN_PATH_PARTS) for part in raw.parts):
        raise ValueError("sealed_packet_path_rejected_before_read")
    lexical = ROOT / raw
    allowed = AUDIT_ROOT / "packets"
    if not lexical.as_posix().casefold().startswith(allowed.as_posix().casefold() + "/"):
        raise ValueError("packet_path_outside_new_internal_audit_allowlist")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_relative_to(allowed.resolve(strict=False)) or resolved.suffix.casefold() != ".json":
        raise ValueError("packet_path_resolution_outside_allowlist")
    return resolved


def _no_redirect_opener(*, direct_connection: bool = False):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    handlers = [NoRedirect]
    if direct_connection:
        # This is opt-in because it bypasses ambient proxy configuration.
        handlers.append(urllib.request.ProxyHandler({}))
    handlers.append(urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    return urllib.request.build_opener(*handlers)


def run_batch(*, batch: str, packets: list[dict], packet_file_sha256: str,
              freeze_sha256: str, batch_dir: Path, key: str, opener,
              direct_connection: bool = False,
              base: str = "https://api.deepseek.com",
              clock: Callable[[], float] = time.time) -> dict:
    """Send a new batch once. Existing directory/journal always blocks resumption."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,63}", batch):
        raise ValueError("invalid_batch_name")
    if batch_dir.exists():
        raise RuntimeError("batch_exists_stop_without_retry")
    if not re.fullmatch(r"[0-9a-f]{64}", packet_file_sha256):
        raise ValueError("packet_file_sha256_invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", freeze_sha256):
        raise ValueError("frozen_corpus_hash_invalid")
    if not packets or len({p.get("case_id") for p in packets}) != len(packets):
        raise ValueError("packet_list_empty_or_duplicate_case_id")
    for packet in packets:
        if not isinstance(packet, dict) or not packet.get("case_id"):
            raise ValueError("packet_case_id_required")
        packet_protocol = packet.get("protocol") or {}
        if packet_protocol.get("version") not in (None, "v129-evidence-review-v1",
                                                    "v130-evidence-review-citations-v1"):
            raise ValueError("packet_protocol_mismatch")

    identity = contract_identity()
    payload_hash = _payload_hash(packets)
    history_root = batch_dir.parent
    if history_root.exists():
        for prior_manifest in history_root.glob("*/manifest-private.json"):
            try:
                prior = json.loads(prior_manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise RuntimeError("unreadable_prior_batch_manifest_stop") from None
            if prior.get("payload_sha256") == payload_hash:
                raise RuntimeError("identical_payload_previously_attempted_stop_without_retry")
    journal = batch_dir / "request-journal-private.jsonl"
    raw_dir = batch_dir / "responses-private"
    raw_dir.mkdir(parents=True)
    manifest = {"version": "v130-deepseek-batch-v1", "batch": batch,
                "packet_file_sha256": packet_file_sha256, "payload_sha256": payload_hash,
                "freeze_sha256": freeze_sha256, "protocol": identity,
                "model": MODEL, "endpoint_host": "api.deepseek.com",
                "request_profile": {"thinking": REQUEST_THINKING,
                                     "max_tokens": REQUEST_MAX_TOKENS,
                                     "response_format": "json_object",
                                     "transport": "direct" if direct_connection else "configured_proxy"},
                "packet_count": len(packets),
                "technical_correction_limit": MAX_TECHNICAL_CORRECTIONS,
                "review_attempted": True, "training_allowed": False}
    (batch_dir / "manifest-private.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _append(journal, {"batch": batch, "payload_sha256": payload_hash,
                      "protocol": identity, "model": MODEL, "event": "batch_started",
                      "time": int(clock()), "training_allowed": False})

    schema = {"reviews": [{"case_id": "exact packet case_id", "approved": True,
             "reason": "...", "risk_labels": [], "missing_fact": "...",
             "search_needed": "yes|no|not_available",
             "query_grounded": "yes|no|not_available",
             "retrieval_relevant": "yes|no|not_available",
             "citations": [{"stage": "question|before_action|retrieved_after_search",
                            "step": None, "document_id": "exact id or null",
                            "quote": "exact substring"}]}]}
    correction_reason = None
    for attempt in (1, 2):
        user_payload = {"packets": packets, "required_output_schema": schema}
        if correction_reason:
            user_payload["technical_correction"] = (
                "Correct only the mechanical validation error: " + correction_reason +
                ". Preserve every semantic decision and rationale. This is the only allowed correction.")
        body_object = {"model": MODEL, "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False,
                                                       separators=(",", ":"))}],
            "temperature": 0, "max_tokens": REQUEST_MAX_TOKENS,
            "thinking": REQUEST_THINKING,
            "response_format": {"type": "json_object"}}
        body = json.dumps(body_object, ensure_ascii=False, separators=(",", ":")).encode()
        request_hash = hashlib.sha256(body).hexdigest()
        _append(journal, {"batch": batch, "attempt": attempt, "request_sha256": request_hash,
                          "event": "sending", "time": int(clock()),
                          "technical_correction": attempt == 2})
        request = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
            method="POST", headers={"Authorization": "Bearer " + key,
                                     "Content-Type": "application/json"})
        try:
            response = opener.open(request, timeout=240)
            raw = response.read()
            status = response.status
        except urllib.error.HTTPError as error:
            raw = error.read()
            status = error.code
        except Exception as error:
            _append(journal, {"batch": batch, "attempt": attempt,
                              "event": "outcome_unknown_stop_without_retry",
                              "error_type": type(error).__name__,
                              "error_errno": getattr(error, "errno", None),
                              "time": int(clock())})
            raise RuntimeError("outcome_unknown_stop_without_retry") from None

        raw_path = raw_dir / f"attempt-{attempt:02d}-raw.json"
        raw_path.write_bytes(raw)
        _append(journal, {"batch": batch, "attempt": attempt,
                          "event": "response_received", "http_status": status,
                          "response_sha256": hashlib.sha256(raw).hexdigest(),
                          "time": int(clock())})
        if not 200 <= status < 300:
            _append(journal, {"batch": batch, "attempt": attempt,
                              "event": "http_error_stop_without_retry", "time": int(clock())})
            raise RuntimeError("http_error_response_received_stop_without_retry")
        try:
            outer = json.loads(raw)
            choice = outer["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length" or not isinstance(content, str) or not content.strip():
                _append(journal, {"batch": batch, "attempt": attempt,
                                  "event": "empty_or_truncated_completion_stop_without_retry",
                                  "finish_reason": finish_reason,
                                  "content_length": len(content) if isinstance(content, str) else None,
                                  "time": int(clock())})
                raise RuntimeError("empty_or_truncated_completion_stop_without_retry")
            decision = json.loads(content)
            by_id = {p["case_id"]: p for p in packets}
            if len(decision.get("reviews", [])) != len(by_id):
                raise ValueError("review_count_mismatch")
            if {v.get("case_id") for v in decision["reviews"]} != set(by_id):
                raise ValueError("review_case_ids_mismatch")
            for vote in decision["reviews"]:
                valid, reason = valid_vote(by_id[vote["case_id"]], vote)
                if not valid:
                    raise ValueError(reason)
        except RuntimeError:
            raise
        except Exception as error:
            correction_reason = type(error).__name__ + ":" + str(error)[:160]
            if attempt == 1:
                _append(journal, {"batch": batch, "attempt": attempt,
                                  "event": "technical_validation_failed_one_correction_allowed",
                                  "reason": correction_reason})
                continue
            _append(journal, {"batch": batch, "attempt": attempt,
                              "event": "invalid_after_technical_correction_stop",
                              "reason": correction_reason})
            raise RuntimeError("invalid_after_technical_correction_stop") from None

        result_path = batch_dir / "validated-review-private.json"
        result = {"version": "v130-deepseek-votes-v1", "batch": batch,
                  "payload_sha256": payload_hash, "protocol": identity,
                  "model": MODEL, "reviews": decision["reviews"],
                  "training_allowed": False}
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        _append(journal, {"batch": batch, "attempt": attempt, "event": "valid_vote_saved",
                          "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                          "time": int(clock()), "training_allowed": False})
        return result
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--send", action="store_true", help="explicitly permit one review API batch")
    parser.add_argument("--direct-connection", action="store_true",
                        help="opt in to direct TLS after diagnosing the configured proxy; does not change system proxy settings")
    args = parser.parse_args()
    packet_path = checked_packet_path(args.packet)
    if not args.send:
        print(json.dumps({"status": "not_sent", "reason": "pass --send only after freeze and gates",
                          "training_allowed": False}))
        return
    bundle = json.loads(packet_path.read_text(encoding="utf-8"))
    if bundle.get("training_allowed") is not False or bundle.get("review_authorized") is not True:
        raise ValueError("packet_bundle_not_authorized_or_training_flag_invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", str(bundle.get("freeze_sha256", ""))):
        raise ValueError("packet_bundle_missing_frozen_corpus_binding")
    if bundle.get("protocol") != contract_identity():
        raise ValueError("packet_bundle_protocol_identity_mismatch")
    if bundle.get("payload_sha256") != _payload_hash(bundle.get("packets", [])):
        raise ValueError("packet_bundle_payload_hash_mismatch")
    key, configured = _credentials()
    if not key or configured != "https://api.deepseek.com":
        raise RuntimeError("approved_deepseek_credentials_or_host_missing")
    batch_dir = AUDIT_ROOT / "deepseek" / args.batch
    result = run_batch(batch=args.batch, packets=bundle["packets"],
        packet_file_sha256=hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        freeze_sha256=bundle["freeze_sha256"],
        batch_dir=batch_dir, key=key,
        opener=_no_redirect_opener(direct_connection=args.direct_connection),
        direct_connection=args.direct_connection)
    print(json.dumps({"status": "valid_vote_saved", "batch": args.batch,
                      "states": len(result["reviews"]), "training_allowed": False}))


if __name__ == "__main__":
    main()
