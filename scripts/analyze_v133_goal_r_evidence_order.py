"""CPU-only Goal R score and order audit with training curves."""
import json
import statistics
from pathlib import Path

from evaluate_v133_goal_c_sft import normalize_answer, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-r-evidence-order"
CASES = ROOT / "runs/v133-goal-p-system/eval-only-cases.jsonl"


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    gold = {x["case_id"]: x for x in rows(CASES) if x["split"] != "train"}
    data = {}
    for label in ("incumbent", "a", "b"):
        part = rows(OUT / f"eval-{label}.jsonl")
        if len(part) != 32 or {x["case_id"] for x in part} != set(gold):
            raise RuntimeError(f"incomplete product evaluation {label}")
        data[label] = {x["case_id"]: x for x in part}
        for x in part:
            actual = gold[x["case_id"]]["official_answer"]
            answer = x["answer"] or ""
            em = int(bool(answer) and normalize_answer(answer) == normalize_answer(actual))
            f1 = f1_score(answer, actual) if answer else 0.
            observed = {d["id"] for step in x["trace"] for d in step.get("observation", {}).get("result", [])}
            if x["em"] != em or abs(x["f1"] - f1) > 1e-9:
                raise RuntimeError(f"score mismatch {label} {x['case_id']}")
            if x["both_support_retrieved"] != set(gold[x["case_id"]]["support_ids"]).issubset(observed):
                raise RuntimeError(f"support mismatch {label} {x['case_id']}")
    diagnostic = rows(OUT / "order-diagnostic.jsonl")
    if len(diagnostic) != 192: raise RuntimeError("order diagnostic incomplete")
    order = {}
    for label in data:
        orig = {x["case_id"]: x for x in diagnostic if x["model"] == label and x["mode"] == "original"}
        reverse = {x["case_id"]: x for x in diagnostic if x["model"] == label and x["mode"] == "reversed"}
        if set(orig) != set(gold) or set(reverse) != set(gold):
            raise RuntimeError(f"diagnostic incomplete {label}")
        if any(orig[k]["em"] != data[label][k]["em"] for k in gold):
            raise RuntimeError(f"original order differs from product result {label}")
        for subset in (orig, reverse):
            for case_id, item in subset.items():
                actual = gold[case_id]["official_answer"]
                em = int(bool(item["answer"]) and normalize_answer(item["answer"]) == normalize_answer(actual))
                if item["em"] != em: raise RuntimeError(f"diagnostic score mismatch {label} {case_id}")
        order[label] = {"original_em": sum(x["em"] for x in orig.values()),
                        "reversed_em": sum(x["em"] for x in reverse.values()),
                        "changed_answers": sum(normalize_answer(orig[k]["answer"] or "") !=
                                               normalize_answer(reverse[k]["answer"] or "") for k in gold),
                        "gained_when_reversed": [k for k in gold if reverse[k]["em"] and not orig[k]["em"]],
                        "lost_when_reversed": [k for k in gold if orig[k]["em"] and not reverse[k]["em"]]}
    paired = {}
    baseline = data["incumbent"]
    for label in ("a", "b"):
        current = data[label]
        paired[label] = {"gained": [k for k in gold if current[k]["em"] and not baseline[k]["em"]],
                         "lost": [k for k in gold if baseline[k]["em"] and not current[k]["em"]]}
    training = {}
    for label in ("a", "b"):
        updates = rows(OUT / label / "updates.jsonl")
        if len(updates) != 64: raise RuntimeError(f"update count mismatch {label}")
        training[label] = {"updates": len(updates), "first_loss": updates[0]["loss"],
                           "last_loss": updates[-1]["loss"],
                           "mean_loss_first_pass": statistics.mean(x["loss"] for x in updates[:32]),
                           "mean_loss_second_pass": statistics.mean(x["loss"] for x in updates[32:])}
    report = {"score_recomputed": True, "support_recomputed": True,
              "product_rows": 96, "diagnostic_rows": 192,
              "paired_vs_incumbent": paired, "order": order, "training": training}
    (OUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3.8))
        for label, color in (("a", "#1565c0"), ("b", "#2e7d32")):
            updates = rows(OUT / label / "updates.jsonl")
            ax.plot(range(1, len(updates) + 1), [x["loss"] for x in updates],
                    label=label.upper(), color=color, alpha=.8)
        ax.set(xlabel="Update", ylabel="Final-answer token CE", title="Answer SFT from the same search model")
        ax.legend(); fig.tight_layout()
        fig.savefig(OUT / "training-curves.png", dpi=160)
        plt.close(fig)
    except ImportError:
        report["plot"] = "matplotlib_unavailable"
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__": main()
