"""Citation-explicit successor contract for new v130 DeepSeek review packets.

Historical v129 packets and votes remain bound to their original contract.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.v128_review_protocol import validate as validate_v128
from scripts.v129_review_protocol import SYSTEM as V129_SYSTEM

PROTOCOL_VERSION = "v130-evidence-review-citations-v1"
SYSTEM = V129_SYSTEM + """

Citation formatting rules (follow these exactly; citations are evidence, not prose references):
- For an answer action, cite only evidence visible in before_action documents. Never cite the
  question itself or retrieved_after_search for an answer. Use the exact document id.
- For a search action, cite the relevant before_action document(s); if there are no prior
  documents, a question citation may quote an exact substring of the question. If retrieved
  results are present, cite the relevant retrieved_after_search document(s) by exact id.
- Every document quote must be a short, contiguous, verbatim substring copied from that one
  document's text. Do not paraphrase, normalize punctuation, repair encoding, or copy a long
  passage containing characters you cannot reproduce exactly. Prefer a short exact substring.
- Never put question text in a document citation, and never cite one document id for text from
  another document. If no suitable exact supporting quote exists, return an empty citations list
  and explain the evidence gap; do not invent or approximate a quote.
"""


def contract_identity() -> dict[str, str]:
    return {
        "version": PROTOCOL_VERSION,
        "system_sha256": hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest(),
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def valid_vote(packet: dict, decision: dict | None) -> tuple[bool, str]:
    if not isinstance(decision, dict):
        return False, "missing_or_malformed_vote"
    try:
        validate_v128(packet, decision)
        # v129's extra temporal citations for rejected searches remain mandatory.
        if packet.get("proposed_action", {}).get("action") == "search":
            from scripts.v129_review_protocol import validate as validate_v129
            validate_v129(packet, decision)
    except (KeyError, TypeError, ValueError) as error:
        return False, str(error)
    return True, "valid_under_v130_semantics_still_need_independent_review"
