from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from agent.llm_client import OpenAICompatibleClient, compact_json
from agent.perception import DatasetProfile


ALLOWED_TOOL_DESCRIPTIONS = {
    "dataset_summary": "Compute numeric summary statistics and basic column overview.",
    "missing_value_check": "Report missing value counts and percentages by column.",
    "correlation_analysis": "Compute numeric correlations and draw a correlation heatmap.",
    "group_comparison": "Compare a numeric metric across a categorical field.",
    "trend_analysis": "Analyze a numeric metric over a date-like field.",
    "chart_generation": "Create a simple chart for the most useful numeric or categorical field.",
}

ALLOWED_TOOLS = set(ALLOWED_TOOL_DESCRIPTIONS)


@dataclass
class PlanStep:
    tool_name: str
    reason: str
    expected_insight: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlanResult:
    steps: list[PlanStep]
    source: str
    error: str | None = None
    raw_response: str | None = None

    def to_dict(self) -> dict:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "source": self.source,
            "error": self.error,
            "raw_response": self.raw_response,
        }


def plan_analysis(
    goal: str,
    profile: DatasetProfile,
    tool_scores: dict[str, float],
    llm_client: OpenAICompatibleClient | None = None,
) -> PlanResult:
    if llm_client and llm_client.ready:
        try:
            raw_response = _request_llm_plan(goal, profile, tool_scores, llm_client)
            steps = parse_llm_plan(raw_response)
            return PlanResult(steps=steps, source="llm", raw_response=raw_response)
        except Exception as exc:
            fallback = fallback_plan(profile, tool_scores)
            return PlanResult(steps=fallback, source="fallback", error=str(exc))

    return PlanResult(steps=fallback_plan(profile, tool_scores), source="fallback")


def parse_llm_plan(raw_response: str) -> list[PlanStep]:
    payload = json.loads(_extract_json(raw_response))
    items = payload.get("tools") or payload.get("actions") or payload.get("plan")
    if not isinstance(items, list) or not items:
        raise ValueError("LLM plan must contain a non-empty tools list.")

    steps: list[PlanStep] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each LLM plan item must be an object.")
        tool_name = item.get("tool_name") or item.get("name") or item.get("tool")
        if tool_name not in ALLOWED_TOOLS:
            raise ValueError(f"LLM selected unknown tool: {tool_name}")
        steps.append(
            PlanStep(
                tool_name=tool_name,
                reason=str(item.get("reason", "Selected by LLM planner.")),
                expected_insight=str(item.get("expected_insight", "Useful dataset insight.")),
            )
        )

    return _deduplicate_steps(steps)


def fallback_plan(profile: DatasetProfile, tool_scores: dict[str, float] | None = None) -> list[PlanStep]:
    scores = tool_scores or {}
    candidates: list[PlanStep] = [
        PlanStep(
            "dataset_summary",
            "The agent needs baseline descriptive statistics before deeper analysis.",
            "Basic distribution and scale of numeric fields.",
        )
    ]

    if any(count > 0 for count in profile.missing_values.values()):
        candidates.append(
            PlanStep(
                "missing_value_check",
                "The dataset contains missing values that may affect analysis quality.",
                "Data quality risks by column.",
            )
        )
    if len(profile.numeric_columns) >= 2:
        candidates.append(
            PlanStep(
                "correlation_analysis",
                "The dataset has multiple numeric columns, so relationships can be estimated.",
                "Strong positive or negative relationships between numeric fields.",
            )
        )
    if profile.categorical_columns and profile.numeric_columns:
        candidates.append(
            PlanStep(
                "group_comparison",
                "The dataset has categorical and numeric fields, so group-level differences are useful.",
                "Which group has the highest average metric.",
            )
        )
    if profile.date_columns and profile.numeric_columns:
        candidates.append(
            PlanStep(
                "trend_analysis",
                "The dataset has a date-like field and numeric metric, so trend analysis is possible.",
                "Whether the selected metric increases or decreases over time.",
            )
        )

    candidates.append(
        PlanStep(
            "chart_generation",
            "A visual check makes the analysis easier to inspect in the demo and report.",
            "A readable distribution or count chart.",
        )
    )

    head = candidates[:1]
    tail = sorted(candidates[1:], key=lambda step: scores.get(step.tool_name, 1.0), reverse=True)
    return _deduplicate_steps(head + tail)


def _request_llm_plan(
    goal: str,
    profile: DatasetProfile,
    tool_scores: dict[str, float],
    llm_client: OpenAICompatibleClient,
) -> str:
    system = (
        "You are a data analysis planning agent. Select only from the allowed tools. "
        "Do not write Python code. Return only JSON."
    )
    user = {
        "goal": goal,
        "dataset_profile": profile.to_dict(),
        "allowed_tools": ALLOWED_TOOL_DESCRIPTIONS,
        "feedback_scores": tool_scores,
        "required_schema": {
            "tools": [
                {
                    "tool_name": "one allowed tool name",
                    "reason": "why this action fits the observed dataset state",
                    "expected_insight": "what the tool should reveal",
                }
            ]
        },
    }
    return llm_client.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": compact_json(user)},
        ],
        temperature=0.1,
    )


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("{"):
        return cleaned
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain JSON.")
    return match.group(0)


def _deduplicate_steps(steps: list[PlanStep]) -> list[PlanStep]:
    seen: set[str] = set()
    result: list[PlanStep] = []
    for step in steps:
        if step.tool_name in seen:
            continue
        seen.add(step.tool_name)
        result.append(step)
    return result

