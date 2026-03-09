from __future__ import annotations

import re
import unicodedata
from typing import Any


_ANSWER_KEYS = (
    "expected_answer",
    "answer",
    "expected_answers",
    "gold_answer",
    "golden_answers",
    "target",
    "label",
)


def _extract_tag(text: str, tag: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _normalize_text(text: Any) -> str:
    raw = "" if text is None else str(text)
    normalized = unicodedata.normalize("NFKD", raw)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.strip().lower()
    return re.sub(r"\s+", " ", normalized)


def _expected_candidates(metadata: dict[str, Any]) -> list[str]:
    expected = metadata.get("expected")
    if expected is None:
        prompt_row = metadata.get("prompt_row")
        if isinstance(prompt_row, dict):
            for key in _ANSWER_KEYS:
                if key in prompt_row and prompt_row[key] is not None:
                    expected = prompt_row[key]
                    break

    if expected is None:
        return []
    if isinstance(expected, (list, tuple)):
        return [candidate for candidate in (_normalize_text(item) for item in expected) if candidate]
    candidate = _normalize_text(expected)
    return [candidate] if candidate else []


def _is_correct(predicted: str, expected_candidates: list[str]) -> bool:
    normalized_predicted = _normalize_text(predicted)
    if not normalized_predicted or not expected_candidates:
        return False
    return any(
        normalized_predicted == candidate
        or normalized_predicted in candidate
        or candidate in normalized_predicted
        for candidate in expected_candidates
    )


def compute_reward(trajectory: Any) -> dict[str, Any]:
    answer_turns = trajectory.agent_trajectories.get("answerer", [])
    predicted = ""
    if answer_turns:
        last_response = getattr(answer_turns[-1], "response_text", "")
        predicted = _extract_tag(last_response, "answer") or last_response

    metadata = getattr(trajectory, "metadata", {}) or {}
    expected_candidates = _expected_candidates(metadata if isinstance(metadata, dict) else {})
    final_reward = 1.0 if _is_correct(predicted, expected_candidates) else 0.0

    return {
        "agent_rewards": {
            role: final_reward for role in trajectory.agent_trajectories
        },
        "final_reward": final_reward,
    }
