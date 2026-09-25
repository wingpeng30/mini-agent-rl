import json

from scripts.audit_v121_arms_api import build_packets, make_trace


def _row(task_id, output, evidence):
    return {"task_id": task_id, "candidate_em": 1, "candidate_f1": 1.0,
            "baseline_em": 1, "calls": [{"action": "answer", "raw_output": output,
                "visible_evidence": [{"id": "doc-1", "title": "Title", "text": evidence}]}],
            "search_queries": [], "retrieved_docs": [], "candidate_answer": output}


def test_blind_packet_omits_gold_scores_ids_and_arm_names():
    task_ids = [f"private-task-id-{i}" for i in range(50)]
    reports = {
        "rows": [_row(task_id, "candidate output", "Exact visible quote") for task_id in task_ids],
    }
    tasks = {task_id: {"question": f"Who founded this {i}?", "answer": "GOLD ANSWER"}
             for i, task_id in enumerate(task_ids)}
    packets = build_packets(reports, reports, tasks)
    serialized = json.dumps(packets, ensure_ascii=False)
    assert len(packets) == 50
    assert "GOLD ANSWER" not in serialized
    assert not any(task_id in serialized for task_id in task_ids)
    assert "candidate_em" not in serialized and "baseline_em" not in serialized
    assert '"A"' not in serialized and '"B"' not in serialized
    assert "Exact visible quote" in serialized


def test_search_gain_evidence_is_kept_after_the_originating_decision():
    row = {
        "calls": [
            {"role": "policy_action", "action": "search",
             "raw_output": '{"action":"search","query":"Stacey King titles"}',
             "visible_evidence": [{"id": "d1", "title": "First", "text": "Stacey King played basketball."}]},
            {"role": "policy_action", "action": "answer",
             "raw_output": '{"action":"answer","answer":"two titles"}',
             "visible_evidence": [{"id": "d1", "title": "First", "text": "Stacey King played basketball."},
                                  {"id": "d2", "title": "Second", "text": "Stacey King won two titles."}]},
        ],
        "retrieved_docs": [
            {"query": "shared initial", "documents": [{"id": "d1", "title": "First", "text": "Stacey King played basketball."}]},
            {"query": "Stacey King titles", "documents": [{"id": "d2", "title": "Second", "text": "Stacey King won two titles."}]},
        ],
        "candidate_answer": "two titles",
    }
    trace = make_trace(row, "trace-x")
    assert "d2" not in [e["document_id"] for e in trace["decisions"][0]["visible_evidence"]]
    assert trace["searches"][1]["originating_decision_index"] == 1
    assert "d2" not in [e["document_id"] for e in trace["searches"][1]["evidence_before"]]
    assert "d2" in [e["document_id"] for e in trace["searches"][1]["evidence_after"]]


def test_anonymous_trace_labels_change_mapping_between_questions():
    task_ids = [f"private-task-{i}" for i in range(50)]
    report_a = {"rows": []}
    report_b = {"rows": []}
    tasks = {}
    for i, task_id in enumerate(task_ids):
        question = f"Question {i}: which person founded the organization?"
        tasks[task_id] = {"question": question, "answer": "hidden target"}
        base = _row(task_id, "candidate", "visible evidence")
        base["retrieved_docs"] = [{"query": f"q{i}-1", "documents": []},
                                  {"query": f"q{i}-2", "documents": []}]
        report_a["rows"].append(base)
        other = _row(task_id, "candidate", "visible evidence")
        other["retrieved_docs"] = [{"query": f"q{i}-1", "documents": []}]
        report_b["rows"].append(other)
    packets = build_packets(report_a, report_b, tasks)
    labels_for_a = {next(t["trace_id"] for t in p["traces"] if len(t["searches"]) == 2) for p in packets}
    assert labels_for_a == {"trace-1", "trace-2"}
