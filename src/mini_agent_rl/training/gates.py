"""共享实验门槛；缺少必需证据时返回 UNVERIFIABLE。"""
from __future__ import annotations
import math
from typing import Any


DEFAULT_LOCAL_MEMORY_LIMIT_GB = 7.4


def resolve_memory_limit_gb(requested_gb: float | None, total_memory_gb: float | None = None) -> float:
    """Resolve a declared memory budget instead of silently retaining an 8GB-era gate.

    A caller may always pin an explicit budget.  Otherwise a visible GPU gets the
    conservative cloud budget required by the 24GB experiment plan; CPU/unit-test
    callers retain the historical safe default.
    """
    if requested_gb is not None:
        if not math.isfinite(requested_gb) or requested_gb <= 0:
            raise ValueError("memory limit must be a finite positive number")
        return float(requested_gb)
    if total_memory_gb is not None and math.isfinite(total_memory_gb) and total_memory_gb > 0:
        return min(21.0, float(total_memory_gb) * 0.9)
    return DEFAULT_LOCAL_MEMORY_LIMIT_GB


def evaluate_probe_gate(*, expected: int, attempted: int, completed: int, failed: int,
                        context: dict[str, Any], memory: dict[str, Any],
                        memory_limit_gb: float | None = None) -> dict[str, Any]:
    limit = resolve_memory_limit_gb(memory_limit_gb, memory.get("total_memory_gb"))
    required = ("lossy_body_trim_rate", "history_drop_rate", "context_budget_errors", "visible_support_retention")
    missing = [key for key in required if context.get(key) is None]
    if memory.get("peak_reserved_gb") is None:
        missing.append("peak_reserved_gb")
    checks = {"all_cases_attempted": attempted == expected, "all_cases_completed": completed == expected,
              "no_failed_cases": failed == 0, "lossy_trim_ok": context.get("lossy_body_trim_rate", 1.0) <= .05,
              "history_drop_ok": context.get("history_drop_rate", 1.0) == 0,
              "budget_error_ok": context.get("context_budget_errors", 1) == 0,
              "support_retention_ok": (context.get("visible_support_retention") or 0) >= .95,
              "memory_ok": memory.get("peak_reserved_gb") is not None and memory["peak_reserved_gb"] <= limit}
    status = "UNVERIFIABLE" if missing else ("PASS" if all(checks.values()) else "FAIL")
    return {"status": status, "missing_fields": missing, "checks": checks,
            "memory_limit_gb": limit,
            "denominators": {"expected": expected, "attempted": attempted, "completed": completed,
                             "failed": failed, "not_run": max(0, expected-attempted)}}
