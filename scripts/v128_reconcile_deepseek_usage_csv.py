"""Reconcile masked daily DeepSeek usage exports with named v124 audit logs.

The API-key column is used in memory only. Neither its value nor a fingerprint
of the secret is written to the report or printed to stdout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from scripts.v124_deepseek_blind_review import ENV_FILE, env_values

ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / "data/evidence-sft-v124-forensic/goal-v128"
JOURNALS = (
    "data/evidence-sft-v124-forensic/repair-deepseek-review-journal-private.jsonl",
    "data/evidence-sft-v124-forensic/repair-v6-deepseek-review-journal-private.jsonl",
    "data/evidence-sft-v124-forensic/repair-v7-deepseek-review-journal-private.jsonl",
    "data/evidence-sft-v124-forensic/repair-v8-deepseek-review-journal-private.jsonl",
    "data/evidence-sft-v124-forensic/pending-dual-review-v9/deepseek-journals/v9-batch-01-journal-private.jsonl",
    "data/evidence-sft-v124-forensic/pending-dual-review-v9/deepseek-journals/v9-batch-01-citation-correction-private.jsonl",
    "data/evidence-sft-v124-forensic/pending-dual-review-v9/deepseek-journals/v9-batch-02-journal-private.jsonl",
    "data/evidence-sft-v124-forensic/pending-dual-review-v9/deepseek-journals/v9-batch-03-journal-private.jsonl",
)
TZ_UTC = timezone.utc
TZ_LOCAL = timezone(timedelta(hours=8))
MASK_CHARS = frozenset("*…•")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], bytes]:
    blob = path.read_bytes()
    text = blob.decode("utf-8-sig")
    return list(csv.DictReader(text.splitlines())), blob


def masked_key_match(local_key: str, masked_value: str) -> dict:
    """Report match facts only; never return key material or a key hash."""
    visible_positions = [i for i, char in enumerate(masked_value)
                         if char not in MASK_CHARS]
    same_length = len(local_key) == len(masked_value)
    visible_match = same_length and all(
        local_key[i] == masked_value[i] for i in visible_positions
    )
    return {
        "csv_key_is_masked": any(char in MASK_CHARS for char in masked_value),
        "key_lengths_equal": same_length,
        "visible_character_count": len(visible_positions),
        "all_visible_characters_match": visible_match,
    }


def request_usage(rows: list[dict[str, str]]) -> list[dict]:
    grouped: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        if row.get("type") != "request_count":
            continue
        key = (row.get("start_time_iso", ""), row.get("api_key_name", ""),
               row.get("model", ""))
        grouped[key] += int(row.get("amount") or 0)
    return [
        {"start_time_iso": day, "api_key_name": name, "model": model,
         "request_count": count}
        for (day, name, model), count in sorted(grouped.items())
    ]


def token_usage(rows: list[dict[str, str]]) -> list[dict]:
    grouped: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        kind = row.get("type", "")
        if kind not in {"input_cache_miss_tokens", "input_cache_hit_tokens",
                        "output_tokens"}:
            continue
        key = (row.get("start_time_iso", ""), row.get("api_key_name", ""), kind)
        grouped[key] += int(row.get("amount") or 0)
    return [
        {"start_time_iso": day, "api_key_name": name,
         "token_type": kind, "tokens": count}
        for (day, name, kind), count in sorted(grouped.items())
    ]


def cost_usage(rows: list[dict[str, str]]) -> list[dict]:
    grouped: dict[tuple[str, str, str], Decimal] = defaultdict(Decimal)
    for row in rows:
        grouped[(row.get("start_time_iso", ""), row.get("wallet_type", ""),
                 row.get("currency", ""))] += Decimal(row.get("cost") or "0")
    return [
        {"start_time_iso": day, "wallet_type": wallet, "currency": currency,
         "cost": format(amount, "f")}
        for (day, wallet, currency), amount in sorted(grouped.items())
    ]


def summarize_journals() -> list[dict]:
    result = []
    for relative in JOURNALS:
        path = ROOT / relative
        if not path.is_file():
            result.append({"path": relative, "present": False})
            continue
        blob = path.read_bytes()
        rows = [json.loads(line) for line in blob.decode("utf-8-sig").splitlines()
                if line.strip()]
        times = []
        for row in rows:
            for key in ("started_unix", "finished_unix"):
                value = row.get(key)
                if isinstance(value, (int, float)):
                    times.append(datetime.fromtimestamp(value, TZ_LOCAL).isoformat())
        request_hashes = sorted({
            row["request_sha256"] for row in rows
            if isinstance(row.get("request_sha256"), str)
            and row["request_sha256"] not in {"", "pending"}
        })
        result.append({
            "path": relative,
            "present": True,
            "sha256": sha256(blob),
            "event_count": len(rows),
            "request_attempt_count_from_unique_hashes": len(request_hashes),
            "first_event_asia_shanghai": min(times) if times else None,
            "last_event_asia_shanghai": max(times) if times else None,
            "models": sorted({r["model"] for r in rows if r.get("model")}),
            "api_base_hosts": sorted({r["api_base_host"] for r in rows
                                       if r.get("api_base_host")}),
            "statuses": [r.get("status") for r in rows if r.get("status")],
        })
    return result


def reconcile(amount_path: Path, cost_path: Path, env_file: Path = ENV_FILE) -> dict:
    amount_rows, amount_blob = read_csv(amount_path)
    cost_rows, cost_blob = read_csv(cost_path)
    local_key = env_values(env_file).get("DEEPSEEK_API_KEY", "").strip()
    amount_names = {}
    for row in amount_rows:
        if row.get("type") == "request_count" and row.get("model") == "deepseek-flash":
            name = row.get("api_key_name", "")
            amount_names[name] = row.get("api_key", "")
    mappings = [
        {"api_key_name": name, **masked_key_match(local_key, masked)}
        for name, masked in sorted(amount_names.items())
    ]
    stat = env_file.stat()
    env_mtime_utc = datetime.fromtimestamp(stat.st_mtime, TZ_UTC).isoformat()
    request_id_columns = {"request_id", "request_uuid", "call_id", "transaction_id"}
    amount_columns = list(amount_rows[0]) if amount_rows else []
    cost_columns = list(cost_rows[0]) if cost_rows else []
    return {
        "version": "v128-deepseek-usage-csv-reconciliation-v1",
        "sources": [
            {"name": amount_path.name, "sha256": sha256(amount_blob),
             "row_count": len(amount_rows),
             "columns_without_secret_field": [c for c in amount_columns if c != "api_key"],
             "request_level_identifiers_present": bool(request_id_columns & set(amount_columns)),
             "api_key_values_persisted": False},
            {"name": cost_path.name, "sha256": sha256(cost_blob),
             "row_count": len(cost_rows),
             "columns": cost_columns,
             "request_level_identifiers_present": bool(request_id_columns & set(cost_columns))},
        ],
        "privacy": {
            "raw_api_key_printed_or_persisted": False,
            "api_key_fingerprint_printed_or_persisted": False,
            "local_env_file_hashed_or_copied": False,
            "matching_method":"compare local key in memory against only visible characters of masked CSV values",
        },
        "local_key_name_matches": mappings,
        "local_env_last_modified_utc": env_mtime_utc,
        "daily_request_counts": request_usage(amount_rows),
        "daily_token_usage": token_usage(amount_rows),
        "daily_costs_account_level": cost_usage(cost_rows),
        "local_review_journals": summarize_journals(),
        "interpretation": {
            "relevant_local_review_window":"2026-09-26 Asia/Shanghai",
            "masked_key_mapping":"All visible CSV key characters match AgentLightning and do not match Tracefix; the exact key value is never recorded.",
            "attribution":"The currently available v124 scripts read the parent .env file; its modification time predates the logged requests. This strongly supports attribution of those script calls to the AgentLightning key label, conditional on those scripts/configuration being the ones used at send time.",
            "request_level_linkage":"The CSVs contain daily aggregates and no per-request ID or timestamp. They corroborate account/day/model activity but cannot identify which individual packet generated a count or determine the outcome of any outcome_unknown request.",
            "cost_scope":"Cost CSV is grouped by day/model/wallet and cannot be assigned to a key label or an individual request.",
            "training_allowed":False,
            "data_ready_for_experiment":False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amount", required=True, type=Path)
    parser.add_argument("--cost", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        default=GOAL / "g3-deepseek-usage-csv-reconciliation-private.json")
    args = parser.parse_args()
    if args.amount.name != "amount-2026-09-15_2026-09-26.csv" or \
            args.cost.name != "cost-2026-09-15_2026-09-26.csv":
        raise ValueError("only_the_user_named_usage_exports_are_allowlisted")
    resolved_output = args.output.resolve()
    if not resolved_output.is_relative_to(GOAL.resolve()):
        raise ValueError("report_output_must_stay_inside_goal_evidence_directory")
    report = reconcile(args.amount, args.cost)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    relevant = [r for r in report["daily_request_counts"]
                if r["start_time_iso"].startswith("2026-09-26")]
    print(json.dumps({"version": report["version"], "sources": report["sources"],
                      "key_name_matches": report["local_key_name_matches"],
                      "relevant_day_request_counts": relevant,
                      "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
