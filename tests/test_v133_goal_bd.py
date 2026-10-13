"""Guards against false acceptance, source drift, and accidental extra experiments."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_v133_goal_bd as bd


def test_split_selection_is_by_identity_not_input_order():
    specs = [dict(split=s, count=n, cases="runs/exposed/"+s+".jsonl", corpus="c.json") for s, n in bd.SPLITS]
    extra = dict(split="ag_regression", count=64, cases="ag.json", corpus="ag-corpus.json")
    assert bd.resolve_specs(dict(evaluation=list(reversed(specs))+[extra])) == specs


def test_forbidden_confirmation_source_rejected():
    specs = [dict(split=s, count=n, cases="runs/exposed/"+s+".jsonl") for s, n in bd.SPLITS]
    specs[0]["cases"] = "runs/unexposed/confirmation-repair-v2/cases.jsonl"
    with pytest.raises(RuntimeError, match="forbidden"):
        bd.resolve_specs(dict(evaluation=specs))


def test_missing_or_unbound_reviews_cannot_pass():
    packets = [dict(arm=a, case_id=str(i), combination_sha256=f"{a}:{i}") for a in ("A", "B") for i in range(29)]
    reviews = [dict(p, category="unconfirmed", reason="existing evidence not sufficient", complete_chain_visible=False, edges=[]) for p in packets]
    assert len(bd.validate_reviews(list(reversed(packets)), reviews)) == 58
    with pytest.raises(RuntimeError, match="58"):
        bd.validate_reviews(packets, reviews[:-1])
    reviews[0]["combination_sha256"] = "other-input"
    with pytest.raises(RuntimeError, match="different actual input"):
        bd.validate_reviews(packets, reviews)


def test_gold_hit_without_edge_proof_cannot_be_success():
    packets = [dict(arm=a, case_id=str(i), combination_sha256=f"{a}:{i}") for a in ("A", "B") for i in range(29)]
    reviews = [dict(p, category="unconfirmed", reason="unchecked", complete_chain_visible=False, edges=[]) for p in packets]
    reviews[0].update(category="success", complete_chain_visible=True)
    with pytest.raises(RuntimeError, match="citations"):
        bd.validate_reviews(packets, reviews)


def test_attempts_consume_budget_even_if_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(bd, "OUT", tmp_path)
    bd.write(tmp_path / "config.json", dict(caps=dict(generation=2), arm_caps=dict(generation=1)))
    bd.write(tmp_path / "usage.jsonl", [], True)
    bd.attempt("generation", "A", "goal_z_dev", "case")
    with pytest.raises(RuntimeError, match="arm quota"):
        bd.attempt("generation", "A", "goal_z_dev", "case")
    assert len(bd.rows(tmp_path / "usage.jsonl")) == 1


def test_final_identity_is_not_best_scoring_seed():
    assert bd.EXPECTED["A"] == "d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52"
    assert len(bd.DEMOS) == 3
    assert sum(n for _, n in bd.SPLITS) == 208
