"""Shared answer parsing / aggregation helpers for `ask` and `force`."""

from __future__ import annotations

import math
import re
from collections import Counter


def normalize_label(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def normalize_response(text: str) -> str:
    """Canonicalize a short response for agreement statistics."""
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized.strip()


def extract_explicit_label(text: str, allowed_labels: list[str]) -> str | None:
    """Return an allowed label if it appears near the front of the response."""
    prefix = " ".join(text.strip().split()[:12])
    allowed = {normalize_label(label): label for label in allowed_labels}
    for match in re.finditer(r"\b[\w-]+\b", prefix.lower()):
        token = match.group(0)
        if token in allowed:
            return allowed[token]
    return None


def is_effectively_empty(text: str) -> bool:
    return not text or not text.strip()


def summarize_responses(
    normalized_responses: list[str],
    *,
    parsed_labels: list[str] | None = None,
    invalid_attempts: int = 0,
) -> dict[str, float | int | str | None]:
    if parsed_labels is not None:
        total = len(parsed_labels)
        counts = Counter(parsed_labels)
        probs = [count / total for count in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        majority_label = None
        majority_label_count = 0
        majority_label_rate = 0.0
        runner_up_label = None
        runner_up_label_count = 0
        if counts:
            ordered = counts.most_common()
            majority_label, majority_label_count = ordered[0]
            majority_label_rate = majority_label_count / total
            if len(ordered) > 1:
                runner_up_label, runner_up_label_count = ordered[1]
        invalid_attempt_rate = invalid_attempts / (total + invalid_attempts) if (total + invalid_attempts) else 0.0
        vote_margin = (
            (majority_label_count - runner_up_label_count) / total
            if total and majority_label_count
            else 0.0
        )
        strength = classify_strength(
            majority_rate=majority_label_rate,
            invalid_attempt_rate=invalid_attempt_rate,
            entropy=entropy,
        )
        return {
            "majority_label": majority_label,
            "majority_label_count": majority_label_count,
            "majority_label_rate": majority_label_rate,
            "runner_up_label": runner_up_label,
            "runner_up_label_count": runner_up_label_count,
            "vote_margin": vote_margin,
            "valid_samples": total,
            "invalid_attempts": invalid_attempts,
            "invalid_attempt_rate": invalid_attempt_rate,
            "unique_labels": len(counts),
            "label_entropy": entropy,
            "strength": strength,
        }

    total = len(normalized_responses)
    counts = Counter(normalized_responses)
    probs = [count / total for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    majority_response = None
    majority_response_count = 0
    majority_response_rate = 0.0
    runner_up_response = None
    runner_up_response_count = 0
    if counts:
        ordered = counts.most_common()
        majority_response, majority_response_count = ordered[0]
        majority_response_rate = majority_response_count / total
        if len(ordered) > 1:
            runner_up_response, runner_up_response_count = ordered[1]
    invalid_attempt_rate = invalid_attempts / (total + invalid_attempts) if (total + invalid_attempts) else 0.0
    vote_margin = (
        (majority_response_count - runner_up_response_count) / total
        if total and majority_response_count
        else 0.0
    )
    strength = classify_strength(
        majority_rate=majority_response_rate,
        invalid_attempt_rate=invalid_attempt_rate,
        entropy=entropy,
    )
    return {
        "majority_response": majority_response,
        "majority_response_count": majority_response_count,
        "majority_response_rate": majority_response_rate,
        "runner_up_response": runner_up_response,
        "runner_up_response_count": runner_up_response_count,
        "vote_margin": vote_margin,
        "valid_samples": total,
        "invalid_attempts": invalid_attempts,
        "invalid_attempt_rate": invalid_attempt_rate,
        "unique_responses": len(counts),
        "response_entropy": entropy,
        "strength": strength,
    }


def classify_strength(*, majority_rate: float, invalid_attempt_rate: float, entropy: float) -> str:
    if majority_rate >= 0.8 and invalid_attempt_rate <= 0.2 and entropy <= 0.75:
        return "strong"
    if majority_rate >= 0.6 and invalid_attempt_rate <= 0.4 and entropy <= 1.25:
        return "mixed"
    return "weak"
