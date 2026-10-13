"""CPU-only, hash-bound Goal K offline attribution and review-packet export."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "runs/v133-goal-j-transfer"
RESULTS = J
G = ROOT / "runs/v133-goal-g-cpu/evaluation-frozen.jsonl"
H_TRAJ = ROOT / "runs/v133-goal-h-comparison/trajectories.jsonl"
OUT = ROOT / "runs/v133-goal-k-attribution"
sys.path.insert(0, str(ROOT))
_scripts_package = types.ModuleType("scripts")
_scripts_package.__path__ = [str(ROOT / "scripts")]
sys.modules["scripts"] = _scripts_package
from scripts.evaluate_v133_goal_c_sft import f1_score, normalize_answer, parse_action
EXPECTED = {
    "train-frozen.jsonl": "d4b41cb3413c9fca736d26b30e6d5e5292c1b47308d21c01f174f8957e0b263b",
    "evaluation-frozen.jsonl": "42531fc1dbd2c3cb1fbec12e85523483c1c0b40546c55a0a6d622474d9648d29",
    "trajectories.jsonl": "ff502d65e2b79d034afc899c743e96276c9767f319175f46ca709b16fa973822",
    "training-config-effective.json": "4ca2cebda898b824c9813c3f424f0181db7d3f21921873711e2dee90fbe43749",
    "training-metrics.json": "2028a7790a0501489b5b4e5968cbe81d4a8c8a820cb68ea84cbb07d06639b394",
    "review-disposition.jsonl": "fdbdbbbb9140857a518518dc96ae439d09c9511f45de0975cf070683bf6d8d42",
    "goal-g-evaluation-frozen.jsonl": "f689f94b598a1c0d89f1b87f0b008c8a03158a7ab479028368285564244aad69",
    "goal-h-trajectories.jsonl": "df247a8b26cdca01aaee4d7d141491fed774ec9840c0e742835ee3721425edab",
    "j-review-packets-1.jsonl": "5881e2faab902d9abe34e8d46cfbbfd4be5cb0ff017aa806eba1c8f7c8410e26",
    "j-review-packets-2.jsonl": "8feb763cc3fb7d378318c82b41808449493f02a9b4d723f2bbecf4ef7e9926c5",
    "j-review-map-1.jsonl": "b57e1649e913ae84e22c3cb58ac41f368c04ff531eb421feb627bac3f9fe78d4",
    "j-review-map-2.jsonl": "9058652b3f5a186658c4e3095e9d85c8b9bc9eee59b6a99517f22e7135b1564a",
    "goal-g-review-packets.jsonl": "c5c44da23a275e76089f67d93b39e25e0e1e9a706628a317007f501aa7e4da26",
    "goal-g-vote-disposition.jsonl": "42df494a9da56035445a8d7e69b94a9bef3f1ca00743a5c0ec1e9a19ee2406d2",
}

# Manual one-pass attribution from each frozen question, complete visible evidence,
# and paired raw outputs. Keys are (split, case_id); no labels are inferred from scores.
MANUAL_ATTRIBUTION = {
    ("train", "5ab563365542992aa134a304"): ("answer_granularity", "从 documentary film 收敛到题目共同类别 documentary。"),
    ("train", "5a8b98c85542995d1e6f1415"): ("invalid_output_recovered", "基线原始输出混有 search 与多段 assistant 文本；新输出是有效的 opera 答案。"),
    ("train", "5a8b8eec55429949d91db5e0"): ("shared_category_selection", "由仅适用于 SWMRS 的 punk rock 改为两者都支持的 band。"),
    ("train", "5ab3a3c4554299233954ff6e"): ("answer_granularity", "将完整句子收敛为证据共同支持的 director。"),
    ("train", "5abe6da155429976d4830b19"): ("shared_category_selection", "将冗长且对一侧过度具体的描述收敛为两侧都支持的 writer。"),
    ("train", "5a90923f5542995651fb5192"): ("unsupported_specificity_removed", "移除两地行政级别相同的错误断言，回答共同的 city。"),
    ("train", "5abadb4355429939ce03dd61"): ("answer_granularity", "将 Josh Trank 的完整职位列表收敛为两侧共同的 director。"),
    ("train", "5abda14f55429965af743d90"): ("answer_timing_and_category", "停止继续检索并回答两份证据均明确支持的 dog。"),
    ("train", "5ab698c75542995eadef002c"): ("answer_granularity", "将解释句收敛为两侧共同的 flowering plant。"),
    ("train", "5adfeffb55429942ec259ba0"): ("requested_attribute_selection", "从单一职业 film director 改为题目所问的共同 industry film。"),
    ("train", "5aba9cff5542994dbf01997e"): ("invalid_output_recovered_and_attribute_selection", "基线原始输出含重复生成片段并选成 Fabaceae；新答案正确指向分类等级 genus。"),
    ("train", "5ae30bfb55429928c42395c7"): ("answer_granularity", "将双方职业列表收敛为证据共同支持的 actor。"),
    ("new_eval", "5a8673585542991e77181662"): ("answer_timing_with_qualifier_omission", "从搜索转为回答 documentary，但省略金标中的 film 限定词，仍未严格答对。"),
    ("new_eval", "5ac00c72554299012d1db589"): ("answer_timing_and_shared_attribute", "从继续搜索转为两份证据共同支持的 novelist，获得严格正确。"),
    ("new_eval", "5ae30de055429928c42395cf"): ("answer_granularity_under_specified", "将完整描述缩为 airport，省略 public use 限定，仍未严格答对。"),
    ("new_eval", "5adde5005542990dbb2f7eee"): ("requested_attribute_selection", "将联赛 Major League Baseball 改为题目询问的运动 baseball，获得严格正确。"),
    ("new_eval", "5ac259bc5542992f1f2b386a"): ("answer_timing_with_qualifier_omission", "从搜索转为 director，但遗漏 film 限定，仍未严格答对。"),
    ("new_eval", "5a78cc3f554299029c4b5e9b"): ("wrong_attribute_and_under_specification", "Pearl Lowe 的 designer 与 Lukas Rossi 的证据不构成共同职业；新答案也漏掉其 singer-songwriter 属性。"),
    ("new_eval", "5ac169c25542994ab5c67d1a"): ("answer_granularity_loss", "director 被两侧证据支持但比问题要求的 mutual specific occupation 更宽，严格EM丢分。"),
    ("goal_G_regression", "g000"): ("answer_timing_and_shared_attribute", "从搜索动作转为正确的共同职业 actress。"),
    ("goal_G_regression", "g001"): ("answer_timing_with_missing_attributes", "从搜索转为 director，但问题问复数职业，漏掉 screenwriter 且未保留 film。"),
    ("goal_G_regression", "g003"): ("answer_granularity", "将仅适用于一侧的 nature 限定移除，得到双方共同的 documentary。"),
    ("goal_G_regression", "g004"): ("answer_granularity_under_specified", "weekly 改为 magazine，但仍漏答两者共同的 women's 限定。"),
    ("goal_G_regression", "g005"): ("shared_attribute_selection", "从只适用于一侧的 harmonica player 改为双方证据支持的 producer。"),
    ("goal_G_regression", "g006"): ("answer_granularity_under_specified", "去掉 rock 后只答 band，未达到金标 rock band 粒度。"),
    ("goal_G_regression", "g008"): ("answer_granularity", "将解释句收敛为共同类别 dog。"),
    ("goal_G_regression", "g009"): ("semantic_equivalent_strict_em_loss", "Democrat 与 Democratic Party 在题目语境指向同一党派；按冻结EM规则仍记丢分。"),
    ("goal_G_regression", "g010"): ("wrong_shared_attribute", "由人物名改为 actress，但仍未回答证据共同支持的 American 国籍属性。"),
    ("goal_G_regression", "g011"): ("answer_granularity", "将两段人物介绍收敛为双方共同的 writer。"),
    ("goal_G_regression", "g014"): ("answer_granularity_under_specified", "从 documentary film series 缩为 documentary，遗漏 film，仍未严格匹配。"),
    ("goal_G_regression", "g015"): ("requested_taxonomic_level_selection", "从 orchid 类群改为问题所问且两份证据支持的 genus。"),
    ("goal_G_regression", "g017"): ("answer_granularity", "将单侧人物的冗长职位列表收敛为两侧共同的 musician。"),
    ("goal_G_regression", "g018"): ("unsupported_shared_attribute", "politician 只由 Tim Kaine 一侧支持；Aneesh Chopra 的证据只说明 executive。"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def action_type(raw: str) -> tuple[str, str | None]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return "unparseable_json", None
    if not isinstance(value, dict) or value.get("action") not in {"answer", "search"}:
        return "invalid_action_shape", None
    kind = value["action"]
    text = value.get("answer" if kind == "answer" else "query")
    if not isinstance(text, str) or not text.strip():
        return "invalid_action_shape", kind
    return kind, text


def check_hashes() -> dict:
    paths = {
        "train-frozen.jsonl": RESULTS / "train-frozen.jsonl",
        "evaluation-frozen.jsonl": RESULTS / "evaluation-frozen.jsonl",
        "trajectories.jsonl": RESULTS / "trajectories.jsonl",
        "training-config-effective.json": RESULTS / "training-config-effective.json",
        "training-metrics.json": RESULTS / "training-metrics.json",
        "review-disposition.jsonl": RESULTS / "review-disposition.jsonl",
        "goal-g-evaluation-frozen.jsonl": G,
        "goal-h-trajectories.jsonl": H_TRAJ,
        "j-review-packets-1.jsonl": J / "review-packets.jsonl",
        "j-review-packets-2.jsonl": J / "review-packets-batch-02.jsonl",
        "j-review-map-1.jsonl": J / "review-private-map.jsonl",
        "j-review-map-2.jsonl": J / "review-private-map-batch-02.jsonl",
        "goal-g-review-packets.jsonl": ROOT / "runs/v133-goal-g-cpu/review-packets.jsonl",
        "goal-g-vote-disposition.jsonl": ROOT / "runs/v133-goal-g-cpu/vote-disposition.jsonl",
    }
    found = {name: sha(path) for name, path in paths.items()}
    mismatches = {name: {"expected": EXPECTED[name], "actual": found[name]}
                  for name in found if found[name] != EXPECTED[name]}
    if mismatches:
        raise ValueError(f"input_hash_mismatch:{json.dumps(mismatches, sort_keys=True)}")
    return {name: {"path": str(paths[name]), "sha256": found[name]} for name in paths}


def score_group(rows: list[dict], split: str, arms: tuple[str, str], source: dict[str, dict]) -> tuple[dict, list[dict]]:
    before_arm, after_arm = arms
    by = {(r["arm"], r["case_id"]): r for r in rows if r["split"] == split}
    cases = list(source)
    if len(cases) != 16 or len(set(cases)) != 16:
        raise ValueError(f"expected_16_unique_cases:{split}")
    paired: list[dict] = []
    for cid in cases:
        before = by.get((before_arm, cid)); after = by.get((after_arm, cid))
        if before is None or after is None:
            raise ValueError(f"missing_arm_case:{split}:{cid}")
        if before["prompt_sha256"] != after["prompt_sha256"]:
            raise ValueError(f"prompt_hash_pair_mismatch:{split}:{cid}")
        btype, bvalue = action_type(before["raw_output"])
        atype, avalue = action_type(after["raw_output"])
        gold = str(before["gold"])
        if gold != str(after["gold"]) or gold != str(source[cid].get("official_answer", source[cid].get("answer", gold))):
            raise ValueError(f"gold_mismatch:{split}:{cid}")
        paired.append({
            "split": split, "case_id": cid, "prompt_sha256": before["prompt_sha256"],
            "gold": gold,
            "before": {"arm": before_arm, "raw_output": before["raw_output"], "action_type": btype,
                       "action_value": bvalue, "prediction": before["prediction"], "protocol_valid": bool(before["protocol_valid"]),
                       "em": int(before["em"]), "f1": float(before["f1"])},
            "after": {"arm": after_arm, "raw_output": after["raw_output"], "action_type": atype,
                      "action_value": avalue, "prediction": after["prediction"], "protocol_valid": bool(after["protocol_valid"]),
                      "em": int(after["em"]), "f1": float(after["f1"])},
            "changed_raw": before["raw_output"] != after["raw_output"],
            "change_tags": [],
            "evidence_attribution": "manual_review_required_for_changed_case",
        })
        tags = paired[-1]["change_tags"]
        if before["raw_output"] != after["raw_output"]:
            tags.append("raw_output_changed")
        if btype == "unparseable_json" or atype == "unparseable_json":
            tags.append("unparseable_json_present")
        elif btype == "invalid_action_shape" or atype == "invalid_action_shape":
            tags.append("invalid_action_shape_present")
        elif btype != atype:
            tags.append(f"action_transition:{btype}_to_{atype}")
        if before["protocol_valid"] != after["protocol_valid"]:
            tags.append("protocol_validity_changed")
        if btype == atype == "answer" and bvalue != avalue:
            tags.append("answer_string_changed")
        if before["em"] != after["em"]:
            tags.append("strict_em_changed")
        if not tags:
            tags.append("no_observed_output_change")
        override = MANUAL_ATTRIBUTION.get((split, cid))
        if override:
            paired[-1]["attribution_class"], paired[-1]["evidence_attribution"] = override
        elif (btype == atype and bvalue == avalue and
              before["protocol_valid"] == after["protocol_valid"] and before["em"] == after["em"]):
            paired[-1]["attribution_class"] = "serialization_only"
            paired[-1]["evidence_attribution"] = "结构化动作和答案值不变；变化仅在原始JSON/空白表示。"
        else:
            paired[-1]["attribution_class"] = "needs_manual_attribution"
            paired[-1]["evidence_attribution"] = "没有预注册映射；停止并登记未决，不猜测归因。"

    def metrics(sub: list[dict], side: str) -> dict:
        key = "before" if side == "before" else "after"
        return {"n": len(sub), "strict_em": sum(x[key]["em"] for x in sub),
                "mean_f1": sum(x[key]["f1"] for x in sub) / len(sub),
                "protocol_valid": sum(x[key]["protocol_valid"] for x in sub)}

    gains = [x["case_id"] for x in paired if not x["before"]["em"] and x["after"]["em"]]
    losses = [x["case_id"] for x in paired if x["before"]["em"] and not x["after"]["em"]]
    retained = [x["case_id"] for x in paired if x["before"]["em"] and x["after"]["em"]]
    both_valid_answer = [x for x in paired if x["before"]["protocol_valid"] and x["after"]["protocol_valid"]
                         and x["before"]["action_type"] == x["after"]["action_type"] == "answer"]
    fixed = {"n": len(both_valid_answer)}
    for side in ("before", "after"):
        fixed[side] = {"strict_em": sum(x[side]["em"] for x in both_valid_answer),
                       "mean_f1": sum(x[side]["f1"] for x in both_valid_answer) / len(both_valid_answer) if both_valid_answer else None}
    summary = {"before": metrics(paired, "before"), "after": metrics(paired, "after"),
               "new_correct": gains, "lost_correct": losses, "retained_correct": retained,
               "fixed_both_valid_answer_subset_posthoc": fixed,
               "fixed_subset_is_primary_metric": False,
               "tag_counts_nonexclusive": dict(Counter(tag for x in paired for tag in x["change_tags"]))}
    return summary, paired


def make_review_packet(row: dict, alias: str, answer: str) -> dict:
    messages = row["messages"]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"expected_target_assistant_at_tail:{alias}")
    visible_messages = messages[:-1]
    before_docs = []
    for msg in visible_messages:
        if msg.get("role") != "tool":
            continue
        try:
            body = json.loads(msg.get("content", ""))
        except json.JSONDecodeError:
            raise ValueError(f"tool_context_not_json:{alias}") from None
        for result in body.get("result", []):
            before_docs.append({"id": result["id"], "title": result.get("title", ""), "text": result["text"]})
    if len(before_docs) != 2:
        raise ValueError(f"expected_two_visible_evidence_docs:{alias}")
    return {
        "case_id": alias,
        "question": next(m["content"] for m in visible_messages if m.get("role") == "user"),
        "visible_messages": visible_messages,
        "before_action": before_docs,
        "retrieved_after_search": [],
        "proposed_action": {"action": "answer", "answer": answer},
        "protocol": {"version": "v130-evidence-review-citations-v1"},
        "review_instructions": "Judge only the candidate answer against the question and complete visible context. Assess if it answers the requested shared property at sufficient granularity, whether each entity is supported by visible evidence, and whether any required qualifier is missing. Cite exact, short text from each supporting document. Do not infer model, run, or condition identity. The official label and historical votes are withheld.",
    }


def run() -> None:
    hashes = check_hashes()
    train = read_jsonl(RESULTS / "train-frozen.jsonl")
    evaluation = read_jsonl(RESULTS / "evaluation-frozen.jsonl")
    goal_g = read_jsonl(G)
    trajectories = read_jsonl(RESULTS / "trajectories.jsonl")
    if len(trajectories) != 96:
        raise ValueError(f"trajectory_count_not_96:{len(trajectories)}")
    train_map = {r["case_id"]: r for r in train}
    eval_map = {r["case_id"]: r for r in evaluation}
    g_map = {r["case_id"]: r for r in goal_g}
    reports = {}
    pair_rows = []
    for split, arms, source in (
        ("train", ("v118", "goal_J_sft"), train_map),
        ("new_eval", ("v118_new_eval", "goal_J_sft"), eval_map),
        ("goal_G_regression", ("v118", "goal_J_sft"), g_map),
    ):
        summary, rows = score_group(trajectories, split, arms, source)
        reports[split] = summary
        pair_rows.extend(rows)
    changed_rows = [r for r in pair_rows if r["changed_raw"]]
    unresolved_attributions = [r["case_id"] for r in changed_rows if r["attribution_class"] == "needs_manual_attribution"]
    if unresolved_attributions or len(MANUAL_ATTRIBUTION) != 33:
        raise ValueError(f"manual_attribution_map_incomplete:{unresolved_attributions}:entries={len(MANUAL_ATTRIBUTION)}")

    # Goal H/J v118 comparison must match all 16 G regression inputs and outputs.
    h = {(x["case_id"]): x for x in read_jsonl(H_TRAJ) if x.get("arm") == "v118" and x.get("split") == "main"}
    g_base = {(x["case_id"]): x for x in trajectories if x.get("arm") == "v118" and x.get("split") == "goal_G_regression"}
    if set(h) != set(g_base) or len(h) != 16:
        raise ValueError("goal_h_goal_j_v118_case_set_mismatch")
    h_matches = []
    for cid in sorted(h):
        fields = ("prompt_sha256", "raw_output", "prediction", "em", "f1", "protocol_valid")
        mismatched = [field for field in fields if h[cid][field] != g_base[cid][field]]
        if mismatched:
            raise ValueError(f"goal_h_goal_j_baseline_mismatch:{cid}:{mismatched}")
        h_matches.append(cid)

    # Static training data-flow contract, bound to effective config and metrics.
    config = json.loads((RESULTS / "training-config-effective.json").read_text(encoding="utf-8"))
    metrics = json.loads((RESULTS / "training-metrics.json").read_text(encoding="utf-8"))
    runner_text = (ROOT / "scripts/run_v133_goal_j_transfer.py").read_text(encoding="utf-8")
    sft_text = (ROOT / "src/mini_agent_rl/training/sft.py").read_text(encoding="utf-8")
    flow = {
        "worker_loads_new_eval_before_training": "manifest,train,new_eval,reg=frozen_inputs()" in runner_text,
        "trainer_input_path_equals_frozen_train": config["trainer_kwargs"]["train_path"].endswith("train-frozen.jsonl"),
        "trainer_validation_path_null": config["trainer_kwargs"].get("validation_path") is None and config.get("validation_path") is None,
        "trainer_evaluation_disabled": config.get("evaluation_enabled") is False and metrics.get("evaluation_enabled") is False,
        "new_eval_path_not_passed_to_trainer": config.get("new_evaluation_path_passed_to_trainer") is False,
        "train_sft_loads_only_named_train_and_optional_validation": "load_jsonl(train_path)" in sft_text and "load_jsonl(validation_path) if validation_path else None" in sft_text,
        "trainer_eval_dataset_none_when_validation_absent": "eval_dataset=validation_data" in sft_text and "validation_data = (Dataset.from_list" in sft_text,
        "checkpoint_metric_selection_present": bool(re.search(r"metric_for_best_model|load_best_model_at_end", sft_text)),
        "trainer_validation_samples": metrics.get("validation_samples"),
    }
    flow["training_dataflow_pass"] = (all(flow[k] is True for k in (
        "trainer_input_path_equals_frozen_train", "trainer_validation_path_null", "trainer_evaluation_disabled",
        "new_eval_path_not_passed_to_trainer", "train_sft_loads_only_named_train_and_optional_validation",
        "trainer_eval_dataset_none_when_validation_absent")) and flow["trainer_validation_samples"] == 0
        and flow["checkpoint_metric_selection_present"] is False)

    # Private provenance map is never included in the DeepSeek or Luna packet.
    loss_specs = [
        ("new_eval", "5ac169c25542994ab5c67d1a", "J-loss-A"),
        ("goal_G_regression", "g009", "J-loss-B"),
    ]
    source_by_split = {"new_eval": eval_map, "goal_G_regression": g_map}
    loss_rows = []
    alias_map = []
    packets = []
    by = {(r["split"], r["arm"], r["case_id"]): r for r in trajectories}
    for split, cid, alias in loss_specs:
        v118_arm = "v118_new_eval" if split == "new_eval" else "v118"
        baseline = by[(split, v118_arm, cid)]
        after = by[(split, "goal_J_sft", cid)]
        packet = make_review_packet(source_by_split[split][cid], alias, str(after["prediction"]))
        packets.append(packet)
        alias_map.append({"alias": alias, "split": split, "case_id": cid,
                          "before_prediction": baseline["prediction"], "candidate_prediction": after["prediction"],
                          "official_answer": source_by_split[split][cid].get("official_answer", baseline["gold"]),
                          "prompt_sha256": after["prompt_sha256"]})
        loss_rows.append(next(x for x in pair_rows if x["split"] == split and x["case_id"] == cid))

    # Bind all strict gains to the exact reviewed question, evidence, target and valid vote pair.
    j_dispositions = {r["source_id"]: r for r in read_jsonl(RESULTS / "review-disposition.jsonl")}
    j_private_maps = read_jsonl(J / "review-private-map.jsonl") + read_jsonl(J / "review-private-map-batch-02.jsonl")
    j_alias_by_source = {r["source_id"]: r["case_id"] for r in j_private_maps}
    j_review_by_alias = {r["case_id"]: r for packet_path in (J / "review-packets.jsonl", J / "review-packets-batch-02.jsonl") for r in read_jsonl(packet_path)}
    g_dispositions = {r["source_id"]: r for r in read_jsonl(ROOT / "runs/v133-goal-g-cpu/vote-disposition.jsonl")}
    g_review_by_case = {r["case_id"]: r for r in read_jsonl(ROOT / "runs/v133-goal-g-cpu/review-packets.jsonl")}
    gains = []
    for split, source in (("train", train_map), ("new_eval", eval_map), ("goal_G_regression", g_map)):
        for cid in reports[split]["new_correct"]:
            frozen = source[cid]
            gain = next(x for x in pair_rows if x["split"] == split and x["case_id"] == cid)
            target = frozen.get("official_answer", gain["gold"])
            if normalize_answer(gain["after"]["prediction"]) != normalize_answer(str(target)):
                raise ValueError(f"gain_answer_not_em_equivalent_to_frozen_target:{split}:{cid}")
            if split in ("train", "new_eval"):
                ticket = j_dispositions.get(frozen["source_id"])
                alias = j_alias_by_source.get(frozen["source_id"])
                packet = j_review_by_alias.get(alias)
                expected_split = "train" if split == "train" else "evaluation"
                if (not ticket or ticket.get("split") != expected_split or ticket.get("disposition") != "qualified" or
                    not ticket.get("deepseek_valid") or not ticket.get("luna_valid") or
                    not ticket.get("deepseek_approved") or not ticket.get("luna_approved") or not packet):
                    raise ValueError(f"goal_j_gain_review_not_bound:{split}:{cid}")
            else:
                ticket = g_dispositions.get(frozen["source_id"])
                packet = g_review_by_case.get(cid)
                if not ticket or ticket.get("disposition") != "qualified" or not ticket.get("deepseek_approved") or not ticket.get("luna_approved") or not ticket.get("cpu_passed") or not packet:
                    raise ValueError(f"goal_g_gain_review_not_bound:{cid}")
            if packet["question"] != next(m["content"] for m in frozen["messages"] if m["role"] == "user"):
                raise ValueError(f"gain_review_question_mismatch:{split}:{cid}")
            docs = {d["id"]: (d.get("title", ""), d["text"]) for d in frozen.get("evidence_documents", [])}
            reviewed_docs = {d["id"]: (d.get("title", ""), d["text"]) for d in packet.get("before_action", [])}
            if docs != reviewed_docs or packet.get("proposed_action", {}).get("answer") != str(target):
                raise ValueError(f"gain_review_context_or_answer_mismatch:{split}:{cid}")
            gains.append({"split": split, "case_id": cid, "source_id": frozen["source_id"],
                          "review_packet_case_id": alias if split in ("train", "new_eval") else cid,
                          "prompt_sha256": gain["prompt_sha256"], "official_answer": target,
                          "prediction": gain["after"]["prediction"],
                          "deepseek_valid_approved": True, "luna_valid_approved": True,
                          "same_frozen_question_context_and_answer": True})

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "input-hashes.json", hashes)
    (OUT / "paired-attribution.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in pair_rows), encoding="utf-8")
    write_json(OUT / "recomputed-metrics.json", reports)
    write_json(OUT / "training-dataflow-audit.json", flow)
    write_json(OUT / "gain-ticket-bindings.json", gains)
    write_json(OUT / "attribution-summary.json", {
        "changed_case_count": len(changed_rows),
        "manually_attributed_case_count": sum(r["attribution_class"] != "serialization_only" for r in changed_rows),
        "serialization_only_case_count": sum(r["attribution_class"] == "serialization_only" for r in changed_rows),
        "nonexclusive_tags": {split: reports[split]["tag_counts_nonexclusive"] for split in reports},
        "fixed_answer_subset_is_posthoc": True,
        "strict_scores_unchanged": True,
    })
    write_json(OUT / "review-private-alias-map.json", alias_map)
    bundle = {"training_allowed": False, "review_authorized": True,
              "freeze_sha256": sha(RESULTS / "evaluation-frozen.jsonl"),
              "protocol": {"version": "v130-evidence-review-citations-v1",
                           "system_sha256": "", "validator_sha256": ""},
              "packets": packets}
    from scripts.v130_review_protocol import contract_identity
    from scripts.v130_deepseek_review import _payload_hash
    bundle["protocol"] = contract_identity()
    bundle["payload_sha256"] = _payload_hash(packets)
    (OUT / "review-packets-anonymous.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "cpu_attribution_ready", "trajectory_rows": len(trajectories),
                      "groups": reports, "goal_h_v118_exact_match_count": len(h_matches),
                      "training_dataflow_pass": flow["training_dataflow_pass"],
                      "review_packet_cases": [p["case_id"] for p in packets],
                      "new_eval_gain_bindings": len(gains)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("run",))
    parser.parse_args()
    run()
