from types import SimpleNamespace

from scripts.evaluate_v121_stop_control import (choose_candidate, generation_context_record,
                                                parse_envelope, summarize, valid_anchor)


def response(content, **updates):
    values = {"parse_error": None, "usage": {}, "tool_calls": [], "content": content}
    values.update(updates)
    return SimpleNamespace(**values)


def test_parse_envelope_reads_client_parsed_answer_payload():
    assert parse_envelope(response('{"status":"missing","anchor":"Atlanta"}')) == {
        "status": "missing", "anchor": "Atlanta"
    }
    assert parse_envelope(response("not-json")) is None
    assert parse_envelope(response('{"status":"complete"}', tool_calls=[object()])) is None


def test_generation_context_skips_post_generation_logprob_replay():
    records = [{"phase": "rollout_sampling", "document_changes": [{"document_id": "d1"}]},
               {"phase": "sampling_old_logprob", "replayed_prompt": True}]
    assert generation_context_record(records, {"phase": "fallback"}) == records[0]
    assert generation_context_record([], {"phase": "fallback"}) == {"phase": "fallback"}


def test_anchor_and_quotes_must_bind_to_question_or_visible_evidence():
    evidence = [{"id": "d1", "text": "Kevin Rathbun ran restaurants in Atlanta."}]
    good = {"anchor": "Atlanta", "anchor_evidence_id": "d1",
            "anchor_quote": "Kevin Rathbun ran restaurants in Atlanta.",
            "facts": [{"fact": "The business was in Atlanta", "evidence_id": "d1",
                       "quote": "Kevin Rathbun ran restaurants in Atlanta."}]}
    assert valid_anchor(good, "What was Atlanta's population?", evidence)
    bad = {**good, "anchor": "Rathbun, Iowa", "anchor_quote": "Kevin Rathbun ran restaurants in Atlanta."}
    assert not valid_anchor(bad, "What was Atlanta's population?", evidence)
    future = {**good, "facts": [{"fact": "Population", "evidence_id": "d1",
                                  "quote": "Atlanta had 470000 people in 2016."}]}
    assert not valid_anchor(future, "What was Atlanta's population?", evidence)


def make_report(correct, v118_ids, v120_ids, f1=0.6, protocol=1.0, searches=60):
    rows = []
    for i in range(50):
        rows.append({"task_id": f"q{i}", "candidate_em": float(i in correct), "candidate_f1": f1,
                     "baseline_em": float(i in v118_ids), "v120_em": float(i in v120_ids),
                     "protocol_valid": i < round(50 * protocol), "termination": "answer",
                     "search_queries": ["q"] * (searches // 50)})
    return {"rows": rows, "metrics": summarize(rows)}


def test_selection_requires_each_preregistered_gate():
    v118 = set(range(14))
    v120 = set(range(26))
    a = make_report(set(range(0, 13)) | set(range(14, 27)), v118, v120)
    b = make_report(set(range(0, 13)) | set(range(14, 30)), v118, v120)
    c = make_report(set(range(29)), v118, v120, f1=0.5)
    d = make_report(set(range(29)), v118, v120, protocol=0.96)
    reports = {"A": a, "B": b, "C": c, "D": d}
    assert choose_candidate(reports) is None  # B fails retention of all 14 v118 correct cases.
    reports["B"] = make_report((v118 | set(range(14, 29))), v118, v120)
    assert choose_candidate(reports) == "B"
