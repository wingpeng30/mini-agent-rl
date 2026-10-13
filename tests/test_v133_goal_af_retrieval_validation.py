import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mini_agent_rl.tools.search import SearchTool
from mini_agent_rl.model import LocalModelConfig
import run_v133_goal_af_retrieval_validation as af


class OffsetTokenizer:
    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        import re
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {"offset_mapping": offsets}


def test_page_paragraph_retrieval_uses_page_title_but_body_only_for_passage_rank():
    corpus = [
        {"id": "p0", "title": "Shared Person", "paragraph_index": 0,
         "text": "Shared Person is a public artist."},
        {"id": "p1", "title": "Shared Person", "paragraph_index": 1,
         "text": "Bibliography"},
        {"id": "p2", "title": "Shared Person", "paragraph_index": 2,
         "text": "Shared Person is also a novelist and editor."},
        {"id": "other", "title": "Shared Person bibliography", "paragraph_index": 0,
         "text": "Bibliography"},
    ]
    tool = SearchTool(corpus, "bm25-v4-page-paragraph", tokenizer=OffsetTokenizer())
    results = tool._bm25_index.search("Shared Person novelist")
    assert results[0]["title"] == "Shared Person"
    assert "novelist" in results[0]["text"]
    assert "Bibliography" not in results[0]["text"]
    assert results[0]["text"].startswith("Shared Person is a public artist.")
    assert tool._bm25_index.last_sources["p2"]["segments"][1]["paragraph_id"] == "p2"


def test_page_paragraph_rejects_missing_tokenizer_and_excludes_pure_section_headings():
    with pytest.raises(ValueError, match="product tokenizer"):
        SearchTool([{"id": "p", "title": "A", "paragraph_index": 0, "text": "A real paragraph."}],
                   "bm25-v4-page-paragraph")
    index = SearchTool([
        {"id": "h", "title": "Alpha", "paragraph_index": 0, "text": "References"},
        {"id": "p", "title": "Alpha", "paragraph_index": 1, "text": "A real paragraph."},
    ], "bm25-v4-page-paragraph", tokenizer=OffsetTokenizer())._bm25_index
    assert "h" not in index.provenance_by_id
    assert "p" in index.provenance_by_id


def test_page_with_no_tokenizable_title_uses_stable_document_identity():
    index = SearchTool([
        {"id": "short-title", "title": "A", "paragraph_index": 0, "text": "Alpha is a real paragraph."},
    ], "bm25-v4-page-paragraph", tokenizer=OffsetTokenizer())._bm25_index
    assert len(index.pages) == 1
    assert "short-title" in next(iter(index.provenance_by_id))


def test_excerpt_window_prefers_query_coverage_then_earliest_tie():
    index = SearchTool([{"id": "p", "title": "Alpha", "paragraph_index": 0, "text": "A paragraph."}],
                       "bm25-v4-page-paragraph", tokenizer=OffsetTokenizer())._bm25_index
    text, span = index._excerpt("red one blue two red three blue four", 3, {"blue"})
    assert text == "red one blue"
    assert span == (0, 12)


def test_goal_af_rejects_wrong_product_adapter_hash_and_release_path():
    release = {"published": True, "selected_route": "trajectory", "selected_seed": 42,
               "adapter": "/x/runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final"}
    assert af.release_identity_matches(release, af.PUBLISHED_ADAPTER_SHA256)
    assert not af.release_identity_matches(release, "e1680c2b2a6a60ec89b4b54249e3132e2005ee6eb94bb04e1acdbd4bf76e064e")
    release["adapter"] = "/x/runs/v133-goal-t-update-length/seed42/update32/adapter-final"
    assert not af.release_identity_matches(release, af.PUBLISHED_ADAPTER_SHA256)


def test_adapter_identity_check_loads_lazy_client_before_inspection(monkeypatch):
    events = []

    class LazyClient:
        _model = None

        def _load(self):
            events.append("load")
            self._model = object()

    def verify(client):
        assert client._model is not None
        events.append("verify")
        return {"ok": True}

    monkeypatch.setattr(af, "verify_loaded_adapter", verify)
    assert af.load_and_verify_adapter(LazyClient()) == {"ok": True}
    assert events == ["load", "verify"]


def test_adapter_identity_uses_peft_canonical_state_not_wrapped_state_dict(monkeypatch):
    model = object()
    expected = {"base_model.model.layer.lora_A.weight": object()}
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        get_peft_model_state_dict=lambda received, adapter_name: expected
        if received is model and adapter_name == "default" else {}))
    assert af.peft_adapter_state(model) is expected


def test_goal_af_approval_is_bound_to_frozen_config_hash():
    assert af.approval_matches("abc123", "abc123")
    assert not af.approval_matches("abc123", "changed")


def test_adapter_base_path_comparison_normalizes_windows_and_linux_paths():
    assert af.base_model_path_matches(r"D:\qwen_08b", Path("/root/autodl-tmp/qwen_08b"))
    assert not af.base_model_path_matches(r"D:\other-model", Path("/root/autodl-tmp/qwen_08b"))


def test_summary_recomputes_em_and_f1_from_frozen_case_labels(monkeypatch):
    monkeypatch.setattr(af.q, "score", lambda pred, gold: {"em": int(pred == gold), "f1": float(pred == gold)})
    rows = [{"case_id": "c1", "answer": "answer", "searches": 2}]
    scored = af.attach_em_f1(rows, {"c1": "answer"})
    assert scored[0]["em"] == 1 and scored[0]["f1"] == 1.0
    assert "em" not in rows[0]
    with pytest.raises(RuntimeError, match="unknown case_id"):
        af.attach_em_f1([{"case_id": "unknown", "answer": "x"}], {})


def test_goal_af_trajectory_ceiling_and_arm_counts_are_fixed():
    assert sum(af.COUNTS.values()) == 208
    assert af.MAX_TRAJECTORIES == 624
    assert af.MAX_MODEL_CALLS == 2496
    assert af.MAX_SEARCHES == 1872


def test_goal_af_can_disable_per_call_timeout_without_changing_default():
    assert LocalModelConfig(timeout=None).timeout is None
    assert LocalModelConfig().timeout == 120.0


def test_goal_af_historical_baseline_gate_stops_on_any_mismatch():
    current = [{"case_id": "x", "answer": "a", "reason": "answer", "searches": 1,
                "model_calls": 2, "trace": [{"prompt_token_ids": [1, 2], "query": "x",
                                               "observation": {"result": [{"id": "d"}]}}]}]
    assert af.compare_historical(current, current)["passed"]
    changed = [{**current[0], "trace": [{**current[0]["trace"][0], "prompt_token_ids": [1, 3]}]}]
    assert not af.compare_historical(changed, current)["passed"]


def test_semantic_review_requires_visible_source_quote_for_success():
    packet = {"r": {"calls": [{"actual_tool_results": [{"id": "doc", "text": "Both are novelists."}],
                                "actual_prompt": "Question. Both are novelists."}]}}
    valid = [{"review_id": "r", "category": "correct_sufficient", "reason": "Both subjects and relation are supported.",
              "first_sufficient_call": 0,
              "diagnostics": [],
              "evidence_quotes": [{"call_index": 0, "source_id": "doc", "text": "Both are novelists."}]}]
    assert af.validate_semantic_reviews(packet, valid)["complete"]
    invalid = [{**valid[0], "evidence_quotes": [{"call_index": 0, "source_id": "doc", "text": "Both are engineers."}]}]
    with pytest.raises(RuntimeError, match="not present"):
        af.validate_semantic_reviews(packet, invalid)
    truncated = {"r": {"calls": [{"actual_tool_results": [{"id": "doc", "text": "Both are novelists."}],
                                   "actual_prompt": "Question only"}]}}
    with pytest.raises(RuntimeError, match="truncated"):
        af.validate_semantic_reviews(truncated, valid)


def test_semantic_review_requires_complete_unique_review_ids():
    packet = {"a": {"calls": []}, "b": {"calls": []}}
    partial = [{"review_id": "a", "category": "indeterminate", "reason": "unclear",
                "diagnostics": [], "first_sufficient_call": None}]
    result = af.validate_semantic_reviews(packet, partial)
    assert not result["complete"] and result["missing_count"] == 1
    with pytest.raises(RuntimeError, match="duplicate"):
        af.validate_semantic_reviews(packet, partial * 2)


def test_json_escaped_evidence_is_visible_but_real_removal_is_rejected():
    text = 'He wrote "Common Sense".\nThe source path is C:\\books.'
    doc = {"id": "doc", "text": text}
    packet = {"r": {"calls": [{"actual_tool_results": [doc],
        "actual_prompt": json.dumps({"result": [doc]}, ensure_ascii=False)}]}}
    review = [{"review_id": "r", "category": "correct_sufficient", "reason": "Visible source supports the answer.",
        "first_sufficient_call": 0, "diagnostics": [],
        "evidence_quotes": [{"call_index": 0, "source_id": "doc", "text": text}]}]
    assert text not in packet["r"]["calls"][0]["actual_prompt"]
    assert af.validate_semantic_reviews(packet, review)["complete"]
    packet["r"]["calls"][0]["actual_prompt"] = json.dumps({"result": []})
    with pytest.raises(RuntimeError, match="truncated"):
        af.validate_semantic_reviews(packet, review)
