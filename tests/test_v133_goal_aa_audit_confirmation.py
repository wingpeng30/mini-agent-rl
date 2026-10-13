"""Goal AA audit and confirmation access invariants."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import run_v133_goal_aa_audit_confirmation as aa
import demo_v133_goal_z_selected as demo


def test_development_packet_covers_each_archived_output():
    packet = aa.z.rows(aa.OUT / "development-packet.jsonl")
    mapping = aa.z.read(aa.OUT / "development-map.json")
    assert len(mapping) == 720
    assert len({x["review_id"] for x in packet}) == len(packet) == 334
    assert {x["review_id"] for x in mapping} == {x["review_id"] for x in packet}
    assert all("model" not in x and "case_id" not in x for x in packet)
    assert all("actual_prompt" in x and "visible_document_ids" in x for x in packet)
    assert {x["split"] for x in mapping} == {"dev", "old", "y"}


def test_renzi_correction_uses_actual_visible_evidence():
    packet = {x["review_id"]: x for x in aa.z.rows(aa.OUT / "development-packet.jsonl")}
    mapping = aa.z.read(aa.OUT / "development-map.json")
    case_id = "5a78bf08554299078472774c"
    by_model = {x["model"]: packet[x["review_id"]] for x in mapping
                if x["split"] == "y" and x["case_id"] == case_id}
    assert len(by_model) == 5
    assert by_model["42-trajectory"]["answer"] == "Italian"
    assert "is an Italian politician" in by_model["42-trajectory"]["actual_prompt"]
    assert "cf86c7ca1bc189d53066" in by_model["42-trajectory"]["visible_document_ids"]
    assert "is an Italian politician" not in by_model["42-answer"]["actual_prompt"]


def test_confirmation_is_closed_until_review_gate():
    with patch.object(aa.z, "read", return_value={"status": "awaiting_development_review"}):
        try:
            aa.evaluate_confirmation()
        except RuntimeError as exc:
            assert "gate closed" in str(exc)
        else:
            raise AssertionError("Confirmation was accessible before review")


def test_published_product_entry_uses_confirmed_seed_42():
    decision = aa.z.read(aa.OUT / "product-decision.json")
    assert decision["published"] is True
    assert decision["selected_seed"] == 42
    label, adapter = demo.selected_adapter()
    assert "Goal AA" in label
    assert adapter == decision["adapter"]
