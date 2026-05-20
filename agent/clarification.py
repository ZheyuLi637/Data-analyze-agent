from __future__ import annotations

import re

from agent.perception import DatasetProfile


VAGUE_ENGLISH = {
    "analyze",
    "analyse",
    "help",
    "insight",
    "insights",
    "pattern",
    "patterns",
    "data",
    "dataset",
    "check",
    "look",
    "find",
    "something",
    "everything",
    "issue",
    "issues",
}

VAGUE_CHINESE_PHRASES = (
    "分析一下",
    "看看",
    "有什么问题",
    "找规律",
    "帮我分析",
)


def goal_is_ambiguous(goal: str) -> bool:
    cleaned = goal.strip().lower()
    if not cleaned:
        return True

    if any(phrase in cleaned for phrase in VAGUE_CHINESE_PHRASES):
        return True

    words = re.findall(r"[a-zA-Z0-9_]+", cleaned)
    if len(words) <= 3:
        return True

    meaningful_words = [word for word in words if word not in VAGUE_ENGLISH]
    return len(meaningful_words) < 2


def suggest_clarifications(profile: DatasetProfile) -> list[str]:
    suggestions: list[str] = []

    missing_columns = [column for column, count in profile.missing_values.items() if count > 0]
    if missing_columns:
        suggestions.append("Audit missing values and explain which columns may reduce analysis reliability.")

    if profile.date_columns and profile.numeric_columns:
        suggestions.append(
            f"Analyze the trend of {profile.numeric_columns[0]} over {profile.date_columns[0]} and identify recent changes."
        )

    if profile.categorical_columns and profile.numeric_columns:
        suggestions.append(
            f"Compare average {profile.numeric_columns[0]} across {profile.categorical_columns[0]} and identify the strongest group."
        )

    if len(profile.numeric_columns) >= 2:
        suggestions.append(
            f"Find relationships between {profile.numeric_columns[0]} and {profile.numeric_columns[1]} and explain possible risks."
        )

    if not suggestions:
        suggestions.append("Summarize the dataset structure and identify the most useful next analysis step.")

    return suggestions[:3]


def clarification_context(goal: str, profile: DatasetProfile) -> dict:
    ambiguous = goal_is_ambiguous(goal)
    suggestions = suggest_clarifications(profile) if ambiguous else []
    return {
        "ambiguous": ambiguous,
        "suggestions": suggestions,
        "planning_goal": _planning_goal(goal, suggestions),
    }


def _planning_goal(goal: str, suggestions: list[str]) -> str:
    if not suggestions:
        return goal
    return f"{goal}\nClarification suggestion used for planning: {suggestions[0]}"
