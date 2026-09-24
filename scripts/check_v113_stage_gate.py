"""Fail closed when v113 P2/P3/P4 prerequisites are absent."""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def decide(stage, source_gate, p3_acceptance=None, p4_acceptance=None):
    required = source_gate.get("required_new_internal_questions", 50)
    eligible = source_gate.get("eligible_new_questions_after_historical_exclusion", 0)
    p2_passed = source_gate.get("passed") is True and eligible >= required
    reasons = []
    if stage in ("p2", "p3", "p4") and not p2_passed:
        reasons.append(f"P2 requires a passed source gate with at least {required} eligible new questions; found {eligible}.")
    if stage in ("p3", "p4") and not (p3_acceptance and p3_acceptance.get("all_internal_gates_passed") is True):
        reasons.append("P3 requires a recorded pass of every frozen internal acceptance gate.")
    if stage == "p4" and not (p4_acceptance and p4_acceptance.get("internal_scheme_frozen") is True):
        reasons.append("P4 requires the complete internal-success scheme to be frozen before independent100 evaluation.")
    return {"stage": stage, "allowed": not reasons, "reasons": reasons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("p2", "p3", "p4"), required=True)
    parser.add_argument("--source-gate", default="reports/v113-p2-source-gate.json")
    parser.add_argument("--p3-acceptance")
    parser.add_argument("--p4-acceptance")
    args = parser.parse_args()
    source_gate = json.loads((ROOT / args.source_gate).read_text(encoding="utf-8"))
    p3 = json.loads((ROOT / args.p3_acceptance).read_text(encoding="utf-8")) if args.p3_acceptance else None
    p4 = json.loads((ROOT / args.p4_acceptance).read_text(encoding="utf-8")) if args.p4_acceptance else None
    result = decide(args.stage, source_gate, p3, p4)
    print(json.dumps(result, ensure_ascii=False))
    if not result["allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
