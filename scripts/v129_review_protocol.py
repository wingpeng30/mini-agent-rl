"""Prospective stricter review validator; v128 packets and votes stay immutable."""
from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.v128_review_protocol import SYSTEM as V128_SYSTEM
from scripts.v128_review_protocol import validate as validate_v128

PROTOCOL_VERSION = "v129-evidence-review-v1"
SYSTEM = V128_SYSTEM + """

For every search decision, including rejection, cite the relevant action-before context when any
exists and cite a retrieved result when results exist. State search_needed, query_grounded, and
retrieval_relevant as yes, no, or not_available. These citation obligations apply regardless of
the approved field value.
"""


def contract_identity() -> dict[str, str]:
    return {
        "version": PROTOCOL_VERSION,
        "system_sha256": hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest(),
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def validate(packet: dict, decision: dict) -> None:
    """Validate v128 bindings plus citation obligations for approved and rejected searches."""
    validate_v128(packet, decision)
    if packet["proposed_action"]["action"] != "search":
        return
    before = packet.get("before_action", [])
    retrieved = packet.get("retrieved_after_search", [])
    citations = decision.get("citations", [])
    stages = {item["stage"] for item in citations}
    expected_prior = "before_action" if before else "question"
    if expected_prior not in stages:
        raise ValueError("search_vote_missing_action_time_prior_citation")
    if retrieved and "retrieved_after_search" not in stages:
        raise ValueError("search_vote_missing_retrieval_citation")
    for key in ("search_needed", "query_grounded", "retrieval_relevant"):
        if decision.get(key) not in {"yes", "no", "not_available"}:
            raise ValueError(f"search_vote_{key}_invalid_or_missing")


def valid_vote(packet: dict, decision: dict | None) -> tuple[bool, str]:
    try:
        if not isinstance(decision, dict):
            return False, "missing_or_malformed_vote"
        validate(packet, decision)
    except (KeyError, TypeError, ValueError) as error:
        return False, str(error)
    return True, "valid_under_v129_semantics_still_need_independent_review"
