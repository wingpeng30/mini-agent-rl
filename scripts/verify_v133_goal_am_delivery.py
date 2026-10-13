"""Read-only experiment validation and final artifact manifest; no model load."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import run_v133_goal_am as am

def main():
    cfg=am.approved()
    p=am.OUT
    assert not am.FAILURE.exists()
    usage=am.rows(p/"gpu-usage.jsonl")
    counts=dict(Counter(r["kind"] for r in usage))
    assert counts==dict(update=256,generation=473,search=257),counts
    summaries={}
    updates={}
    for arm in ("A","B"):
        s=am.read(p/f"seed42-{arm}/summary.json")
        assert s["updates"]==128 and s["base_unchanged"] and s["changed_lora_tensors"]==192 and s["reload_max_abs"]==0
        assert am.sha(p/f"seed42-{arm}/adapter-final/adapter_model.safetensors")==s["adapter_sha256"]
        u=am.rows(p/f"seed42-{arm}/updates.jsonl")
        assert [r["update"] for r in u]==list(range(1,129))
        assert [r["trajectory_id"] for r in u]==cfg["order"]["42"]
        assert [r["update"] for r in u if r["state_id"] is not None]==list(range(2,69,2))
        assert len({r["state_id"] for r in u if r["state_id"] is not None})==34
        assert all(math.isfinite(r[k]) for r in u for k in ("trajectory_loss","state_loss","gradient"))
        updates[arm]=u;summaries[arm]=s
        rr=am.rows(p/f"eval-42-{arm}.jsonl")
        assert len(rr)==64
        for r in rr:
            assert r["model_calls"]<=4 and r["searches"]<=3
            for t in r["trace"]:
                assert len(t["prompt_token_ids"])<=16384 and len(t["action_token_ids"])<=2048
                assert len(t.get("observation",{}).get("result",[]))<=3
        frozen={r["probe_id"]:r for r in am.rows(p/"frozen-probes.jsonl")}
        probes=am.rows(p/f"probe-42-{arm}.jsonl")
        assert len(probes)==32
        assert all(r["prompt_token_ids"]==frozen[r["probe_id"]]["prompt_token_ids"] for r in probes)
    assert [(r["trajectory_id"],r["state_id"]) for r in updates["A"]]==[(r["trajectory_id"],r["state_id"]) for r in updates["B"]]
    states=am.rows(p/"candidate-training-states.jsonl")
    assert len(states)==34
    assert len(am.rows(p/"probe-baseline-state.jsonl"))==24
    gate=am.read(p/"seed42-gate.json")
    assert not gate["route_passed"] and not gate["released"]
    assert all(not x["complete_probe_gain_two"] for x in gate["gates"].values())
    assert not list(p.glob("seed43*"))
    assert len(am.rows(p/"semantic-review-seed42.jsonl"))==58
    assert len(am.rows(p/"probe-review-seed42.jsonl"))==64
    assert len(am.rows(p/"probe-evidence-seed42.jsonl"))==48
    corrected=sum(r["action_label_mechanically_corrected"] for r in am.rows(p/"probe-review-packets-seed42.jsonl"))
    assert corrected==16
    assert len(am.read(p/"citation-revisions-seed42.json"))==8
    previous=am.read(p/"status.json")
    am.write_once(p/"status-before-delivery.json",previous)
    status=dict(status="complete_seed42_gate_failed",successful_optimizer_updates=256,
        gpu_generations=473,real_searches=257,full_search_trajectories=128,training_runs=2,
        seed43_executed=False,released=False,stop_reason="complete evidence probe gain is 1, required 2")
    (p/"status.json").write_text(json.dumps(status,indent=2)+"\n")
    validation=dict(passed=True,freeze_sha256=am.sha(p/"execution-freeze.json"),
        frozen_inputs_unchanged=True,candidate_weights_match=True,matched_update_order=True,
        finite_training_statistics=True,actual_evaluation_token_and_call_limits=True,
        actual_probe_inputs_match=True,counts=counts,full_search_trajectories=128,
        corrected_probe_labels=corrected,corrected_citation_pointers=8,
        semantic_review_records=58,probe_review_records=64,
        complete_probe_evidence_records=48,seed43_executed=False,released=False,
        candidates=summaries,limitations=["probe generated token IDs not archived",
        "8 missing-link probes are initial no-tool-input states", "single seed; no full regression or fresh confirmation"])
    am.write_once(p/"delivery-validation.json",validation)
    files={str(f.relative_to(p)):am.sha(f) for f in sorted(p.rglob("*")) if f.is_file()}
    scripts=["scripts/run_v133_goal_am.py","scripts/analyze_v133_goal_am.py",
        "scripts/finalize_v133_goal_am_seed42.py","scripts/verify_v133_goal_am_delivery.py"]
    am.write_once(p/"delivery-manifest.json",dict(files=files,scripts={s:am.sha(am.ROOT/s) for s in scripts},
        status=status,execution_freeze_sha256=am.sha(p/"execution-freeze.json")))
    print(json.dumps(dict(passed=True,counts=counts,manifest_files=len(files),seed43=False,released=False)))

if __name__=="__main__":main()
