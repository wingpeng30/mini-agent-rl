from scripts.analyze_v103 import classify, paired_bootstrap


def test_classification_uses_actual_tool_messages():
    rollout={"id":"r","task_id":"t","final_answer":"wrong person","termination_reason":"answer"}
    transitions=[{"messages":[
        {"role":"system","content":"p"},
        {"role":"tool","content":"{\"result\":[{\"title\":\"Support\",\"text\":\"The answer is Ada Lovelace.\"}]}"},
    ]}]
    task={"question":"Who?","answer":"Ada Lovelace","metadata":{"support_titles":["Support"]}}
    row=classify(rollout,transitions,task)
    assert row["gold_string_in_actual_prompt"] is True
    assert row["support_titles_complete"] is True
    assert row["category"] == "entity_or_relation_selection_error"


def test_bootstrap_clusters_one_value_per_task():
    result=paired_bootstrap({"a":1.0,"b":-1.0},repeats=1000)
    assert result["n_tasks"] == 2
    assert result["delta"] == 0.0
