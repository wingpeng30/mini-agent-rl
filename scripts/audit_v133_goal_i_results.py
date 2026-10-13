"""Bind Goal I's gains/losses to exact-context approved evidence audits."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.v130_review_protocol import valid_vote

RUN=ROOT/"runs/v133-goal-i-microfit/attempt-02"
G=ROOT/"runs/v133-goal-g-cpu"

def readl(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def valid_legacy_luna(packet,vote):
    docs={d["id"]:d["text"] for d in packet["before_action"]}
    return (vote.get("case_id")==packet["case_id"] and type(vote.get("approved")) is bool
        and bool(vote.get("reason")) and all(c.get("document_id") in docs and c.get("quote") in docs[c["document_id"]]
        for c in vote.get("citations",[])))

summary=json.loads((RUN/"summary.json").read_text(encoding="utf-8"))
assert summary["status"]=="complete_microfit_diagnostic"
traces=readl(RUN/"trajectories.jsonl")
g_eval={r["case_id"]:r for r in readl(G/"evaluation-frozen.jsonl")}
g_packets={r["case_id"]:r for r in readl(G/"review-packets.jsonl")}
g_luna={r["case_id"]:r for r in json.loads((G/"luna-votes-technical-correction.json").read_text(encoding="utf-8"))}
g_disposition={r["case_id"]:r for r in readl(G/"vote-disposition.jsonl")}
deepseek_by_batch={}
for folder in (G/"deepseek").glob("g-*"):
    path=folder/"validated-review-private.json"
    if path.exists():
        blob=json.loads(path.read_text(encoding="utf-8"))
        deepseek_by_batch[blob["batch"]]={r["case_id"]:r for r in blob["reviews"]}

bindings=[]
for case_id in summary["comparisons"]["goal_G_exploratory"]["new_correct"]:
    case=g_eval[case_id]
    packet=g_packets[case_id]
    trace=next(r for r in traces if r["arm"]=="microfit" and r["split"]=="exploratory_eval" and r["case_id"]==case_id)
    batch=case["review_refs"]["deepseek_batch"]
    dv=deepseek_by_batch[batch][case_id]
    lv=g_luna[case_id]
    assert packet["visible_messages"]==case["messages"][:case["target_assistant_index"]]
    assert packet["proposed_action"]==json.loads(case["messages"][-1]["content"])
    assert json.loads(trace["raw_output"])["answer"]==case["official_answer"]
    assert g_disposition[case_id]["disposition"]=="qualified"
    assert valid_vote(packet,dv)[0] and valid_legacy_luna(packet,lv)
    assert lv["approved"] is True
    bindings.append({"case_id":case_id,"kind":"new_strict_EM_correct","same_context_and_answer_as_G_packet":True,
        "deepseek_approved":dv["approved"],"luna_approved":lv["approved"],
        "disposition":g_disposition[case_id]["disposition"],"review_reused":True})

post_packets={r["case_id"]:r for r in readl(RUN/"post-eval-review-packets.jsonl")}
post_luna={r["case_id"]:r for r in json.loads((RUN/"post-eval-luna-votes.json").read_text(encoding="utf-8"))}
post_deepseek=json.loads((RUN/"deepseek/i-regression-01/validated-review-private.json").read_text(encoding="utf-8"))
post_deepseek={r["case_id"]:r for r in post_deepseek["reviews"]}
for original,alias in (("g009","J-A"),("g013","J-B")):
    packet=post_packets[alias]
    trace=next(r for r in traces if r["arm"]=="microfit" and r["split"]=="exploratory_eval" and r["case_id"]==original)
    assert packet["proposed_action"]==json.loads(trace["raw_output"])
    assert packet["visible_messages"]==g_eval[original]["messages"][:g_eval[original]["target_assistant_index"]]
    lv,dv=post_luna[alias],post_deepseek[alias]
    assert valid_vote(packet,lv)[0] and valid_vote(packet,dv)[0]
    bindings.append({"case_id":original,"review_alias":alias,"kind":"strict_EM_loss_candidate_review",
        "proposed_action":packet["proposed_action"],"deepseek_approved":dv["approved"],
        "luna_approved":lv["approved"],"decision":"both reviewers find proposed answer evidence-supported" if lv["approved"] and dv["approved"] else "not jointly approved",
        "review_reused":False})

result={"version":"v133-goal-i-result-review-binding-v1","bindings":bindings,
    "new_correct_cases":summary["comparisons"]["goal_G_exploratory"]["new_correct"],
    "strict_loss_cases":summary["comparisons"]["goal_G_exploratory"]["lost_correct"],
    "strict_loss_semantic_reviews_jointly_approved":all(x["deepseek_approved"] and x["luna_approved"] for x in bindings if x["kind"]=="strict_EM_loss_candidate_review"),
    "goal_G_is_exploratory_only":True,
    "hashes":{"summary":sha(RUN/"summary.json"),"trajectories":sha(RUN/"trajectories.jsonl"),
        "post_eval_packets":sha(RUN/"post-eval-review-packets.jsonl"),
        "post_eval_deepseek":sha(RUN/"deepseek/i-regression-01/validated-review-private.json"),
        "post_eval_luna":sha(RUN/"post-eval-luna-votes.json")}}
(RUN/"result-review-binding.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"audit":"pass","new_correct":result["new_correct_cases"],
    "strict_losses":result["strict_loss_cases"],"loss_answers_supported":result["strict_loss_semantic_reviews_jointly_approved"]},ensure_ascii=False))
