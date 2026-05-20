from __future__ import annotations


DEFAULT_TOOL_SCORES = {
    "dataset_summary": 1.0,
    "missing_value_check": 1.0,
    "correlation_analysis": 1.0,
    "group_comparison": 1.0,
    "trend_analysis": 1.0,
    "chart_generation": 1.0,
}


def initial_tool_scores() -> dict[str, float]:
    return dict(DEFAULT_TOOL_SCORES)


def update_tool_scores(
    scores: dict[str, float],
    tools_used: list[str],
    feedback: str,
) -> dict[str, float]:
    updated = dict(scores)
    if feedback not in {"useful", "not_useful"}:
        return updated

    delta = 0.1 if feedback == "useful" else -0.1
    for tool_name in tools_used:
        current = updated.get(tool_name, 1.0)
        updated[tool_name] = round(max(0.5, current + delta), 2)
    return updated

