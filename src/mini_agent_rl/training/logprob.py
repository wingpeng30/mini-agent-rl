"""v0.6.1：本地策略 action logprob 的稳定性诊断。"""
from __future__ import annotations

import math
from typing import Any


def _percentile(values: list[float], q: float) -> float:
    if not values: return 0.0
    ordered = sorted(values); index = (len(ordered) - 1) * q
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def compare_logprobs(left: list[float], right: list[float]) -> dict[str, Any]:
    """比较两个相同离散 action 的 logprob；长度不符不能被统计量掩盖。"""
    finite = len(left) == len(right) and all(math.isfinite(x) for x in left + right)
    if not finite:
        return {"valid": False, "length_left": len(left), "length_right": len(right), "error": "长度不一致或出现 NaN/Inf"}
    delta = [abs(a - b) for a, b in zip(left, right)]
    ratios = [math.exp(max(-30.0, min(30.0, a - b))) for a, b in zip(left, right)]
    return {"valid": True, "token_count": len(delta), "absolute_delta": {"p50": _percentile(delta, .50), "p95": _percentile(delta, .95), "p99": _percentile(delta, .99), "max": max(delta, default=0.0)},
            "ratio": {"p50": _percentile(ratios, .50), "p95": _percentile(ratios, .95), "p99": _percentile(ratios, .99), "min": min(ratios, default=1.0), "max": max(ratios, default=1.0)}}


def stable_same_instance(scores: list[list[float]]) -> dict[str, Any]:
    """以第一轮为参考；任何 token 边界变化都会明确报错。"""
    comparisons = [compare_logprobs(scores[0], value) for value in scores[1:]] if scores else []
    max_delta = max((item.get("absolute_delta", {}).get("max", float("inf")) for item in comparisons), default=0.0)
    return {"repeats": len(scores), "comparisons": comparisons, "max_absolute_delta": max_delta,
            "passed": bool(comparisons or scores) and all(item["valid"] for item in comparisons) and max_delta <= 1e-6}
