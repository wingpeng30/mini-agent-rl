"""Shared answer normalization and bounded partial-match scoring."""

from __future__ import annotations

from collections import Counter
import re
import string


def normalize_answer(text: str | None) -> str:
    value = (text or "").lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = value.translate(str.maketrans("", "", string.punctuation))
    return " ".join(value.split())


def token_f1(prediction: str | None, reference: str | None) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(reference).split()
    if not pred or not gold:
        return float(not pred and not gold)
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def reward_token_f1(prediction: str | None, reference: str | None) -> float:
    """Hotpot yes/no answers require exact normalized agreement."""
    gold = normalize_answer(reference)
    if gold in {"yes", "no"}:
        return float(normalize_answer(prediction) == gold)
    return token_f1(prediction, reference)
