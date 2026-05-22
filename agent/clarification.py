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

ANALYSIS_INTENT_ENGLISH = {
    "analyze",
    "analyse",
    "audit",
    "check",
    "compare",
    "correlation",
    "explain",
    "find",
    "help",
    "insight",
    "insights",
    "pattern",
    "patterns",
    "quality",
    "relationship",
    "risk",
    "risks",
    "summarize",
    "summary",
    "trend",
    "understand",
}

ANALYSIS_INTENT_CHINESE = (
    "分析",
    "比较",
    "趋势",
    "风险",
    "质量",
    "缺失",
    "关系",
    "总结",
    "看看",
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

    group_columns = _usable_suggestion_group_columns(profile)
    if group_columns and profile.numeric_columns:
        suggestions.append(
            f"Compare average {profile.numeric_columns[0]} across {group_columns[0]} and identify the strongest group."
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
    requires_user_input = ambiguous and not goal_has_analysis_intent(goal, profile)
    suggestions = suggest_clarifications(profile) if ambiguous else []
    return {
        "ambiguous": ambiguous,
        "requires_user_input": requires_user_input,
        "suggestions": suggestions,
        "planning_goal": _planning_goal(goal, suggestions, requires_user_input),
    }


def goal_has_analysis_intent(goal: str, profile: DatasetProfile) -> bool:
    cleaned = goal.strip().lower()
    if not cleaned:
        return False

    if any(phrase in cleaned for phrase in ANALYSIS_INTENT_CHINESE):
        return True

    words = set(re.findall(r"[a-zA-Z0-9_]+", cleaned))
    if words & ANALYSIS_INTENT_ENGLISH:
        return True

    return any(column.lower() in cleaned for column in profile.columns if len(column) > 1)


def _planning_goal(goal: str, suggestions: list[str], requires_user_input: bool) -> str:
    return goal


def _usable_suggestion_group_columns(profile: DatasetProfile) -> list[str]:
    row_count = max(profile.row_count, 1)
    threshold = min(30, max(2, int(row_count * 0.5)))
    return [
        column
        for column in profile.categorical_columns
        if 1 < profile.categorical_cardinality.get(column, 0) <= threshold
        and not (profile.generic_column_names and column.lower().startswith("column_"))
    ]
