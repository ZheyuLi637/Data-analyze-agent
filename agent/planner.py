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
    "date_quality_check": "Check whether date-like columns can be parsed reliably.",
    "text_analysis": "Extract lightweight keywords and sentiment from substantial text columns.",
    "topic_modeling": "Group repeated text terms into recurring themes with example rows.",
    "statistical_testing": "Run bounded statistical tests such as correlation significance or group mean difference checks.",
    "predictive_modeling": "Fit a simple local baseline model for numeric prediction or trend forecasting.",
    "causal_risk_analysis": "Audit observational associations and list causal caveats/confounders without claiming causality.",
    "chart_generation": "Create a simple chart for the most useful numeric or categorical field.",
}

ALLOWED_TOOLS = set(ALLOWED_TOOL_DESCRIPTIONS)
MAX_PLAN_STEPS = 4


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
            steps = filter_applicable_steps(parse_llm_plan(raw_response), profile)
            if not steps:
                raise ValueError("LLM plan did not contain any applicable tools for this dataset.")
            return PlanResult(steps=steps, source="llm", raw_response=raw_response)
        except Exception as exc:
            fallback = fallback_plan(profile, tool_scores, goal)
            return PlanResult(steps=fallback, source="fallback", error=str(exc))

    return PlanResult(steps=fallback_plan(profile, tool_scores, goal), source="fallback")


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

    return _deduplicate_steps(steps)[:MAX_PLAN_STEPS]


def filter_applicable_steps(steps: list[PlanStep], profile: DatasetProfile) -> list[PlanStep]:
    applicable: list[PlanStep] = []
    for step in steps:
        if step.tool_name == "correlation_analysis" and len(profile.numeric_columns) < 2:
            continue
        if step.tool_name == "group_comparison" and not (_usable_group_columns(profile) and profile.numeric_columns):
            continue
        if step.tool_name == "trend_analysis" and not (_reliable_date_columns(profile) and profile.numeric_columns):
            continue
        if step.tool_name == "date_quality_check" and not profile.date_parse_percent:
            continue
        if step.tool_name == "text_analysis" and not profile.text_columns:
            continue
        if step.tool_name == "topic_modeling" and not profile.text_columns:
            continue
        if step.tool_name == "statistical_testing" and not (
            len(profile.numeric_columns) >= 2 or (profile.categorical_columns and profile.numeric_columns)
        ):
            continue
        if step.tool_name == "predictive_modeling" and not profile.numeric_columns:
            continue
        if step.tool_name == "causal_risk_analysis" and not profile.numeric_columns:
            continue
        if step.tool_name == "missing_value_check" and not any(count > 0 for count in profile.missing_values.values()):
            continue
        applicable.append(step)
    return applicable[:MAX_PLAN_STEPS]


def fallback_plan(
    profile: DatasetProfile,
    tool_scores: dict[str, float] | None = None,
    goal: str = "",
) -> list[PlanStep]:
    scores = tool_scores or {}
    candidates = _candidate_steps(profile)
    focused = _focus_steps(candidates, goal)

    if focused:
        return _deduplicate_steps(focused)[:MAX_PLAN_STEPS]

    return _deduplicate_steps(_default_steps(candidates, scores, profile))[:MAX_PLAN_STEPS]


def _candidate_steps(profile: DatasetProfile) -> list[PlanStep]:
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
    if _usable_group_columns(profile) and profile.numeric_columns:
        candidates.append(
            PlanStep(
                "group_comparison",
                "The dataset has categorical and numeric fields, so group-level differences are useful.",
                "Which group has the highest average metric.",
            )
        )
    if _reliable_date_columns(profile) and profile.numeric_columns:
        candidates.append(
            PlanStep(
                "trend_analysis",
                "The dataset has a date-like field and numeric metric, so trend analysis is possible.",
                "Whether the selected metric increases or decreases over time.",
            )
        )
    if profile.date_parse_percent:
        candidates.append(
            PlanStep(
                "date_quality_check",
                "The dataset has date-like fields, so date parse reliability should be checked before trend claims.",
                "Whether time-based analysis is trustworthy.",
            )
        )
    if profile.text_columns:
        candidates.append(
            PlanStep(
                "text_analysis",
                "The dataset is mostly text, so lightweight keyword and sentiment analysis is more useful than numeric tools.",
                "Common themes and approximate sentiment in text rows.",
            )
        )
        candidates.append(
            PlanStep(
                "topic_modeling",
                "The dataset contains text/categorical fields, so recurring text themes can be grouped.",
                "Topic-like clusters of repeated terms with example rows.",
            )
        )

    if len(profile.numeric_columns) >= 2 or (_usable_group_columns(profile) and profile.numeric_columns):
        candidates.append(
            PlanStep(
                "statistical_testing",
                "The dataset supports a bounded significance-style check for relationships or group differences.",
                "Whether the strongest observed pattern is statistically notable.",
            )
        )

    if profile.numeric_columns:
        candidates.append(
            PlanStep(
                "predictive_modeling",
                "The dataset has numeric outcomes, so a simple baseline prediction can be fit locally.",
                "A baseline model fit, error estimate, and next-step prediction.",
            )
        )
        candidates.append(
            PlanStep(
                "causal_risk_analysis",
                "The dataset may suggest causal hypotheses, but observational risks must be audited.",
                "Association evidence plus confounders and causal guardrails.",
            )
        )

    candidates.append(
        PlanStep(
            "chart_generation",
            "A visual check makes the analysis easier to inspect in the demo and report.",
            "A readable distribution or count chart.",
        )
    )

    return candidates


def _reliable_date_columns(profile: DatasetProfile) -> list[str]:
    return [
        column
        for column in profile.date_columns
        if profile.date_parse_percent.get(column, 0) >= 80
    ]


def _usable_group_columns(profile: DatasetProfile) -> list[str]:
    row_count = max(profile.row_count, 1)
    threshold = min(30, max(2, int(row_count * 0.5)))
    return [
        column
        for column in profile.categorical_columns
        if 1 < profile.categorical_cardinality.get(column, 0) <= threshold
        and not (profile.generic_column_names and not _semantic_group_column(column))
    ]


def _semantic_group_column(column: str) -> bool:
    tokens = set(re.split(r"[^a-zA-Z0-9]+", column.lower()))
    return bool(tokens & {"region", "category", "segment", "product", "customer", "group", "type", "channel", "market"})


def _focus_steps(candidates: list[PlanStep], goal: str) -> list[PlanStep]:
    lowered = goal.lower()
    by_tool = {step.tool_name: step for step in candidates}
    selected: list[PlanStep] = []

    def add(tool_name: str) -> None:
        if tool_name in by_tool:
            selected.append(by_tool[tool_name])

    add("dataset_summary")

    if _goal_has_any(lowered, ("quality", "missing", "null", "clean", "audit", "reliability")):
        add("missing_value_check")
        add("date_quality_check")
        add("chart_generation")
    elif _goal_has_any(lowered, ("text", "feedback", "comment", "review", "sentiment", "keyword", "theme", "topic")):
        add("text_analysis")
        add("topic_modeling")
    elif _goal_has_any(lowered, ("predict", "forecast", "model", "estimate", "projection", "future")):
        add("predictive_modeling")
        add("trend_analysis")
        add("chart_generation")
    elif _goal_has_any(lowered, ("significant", "significance", "p-value", "p value", "hypothesis", "test")):
        add("statistical_testing")
        add("correlation_analysis")
        add("group_comparison")
    elif _goal_has_any(lowered, ("causal", "causality", "cause", "impact", "effect", "driver", "why")):
        add("causal_risk_analysis")
        add("statistical_testing")
        add("chart_generation")
    elif _goal_has_any(lowered, ("trend", "time", "date", "over time", "recent")):
        add("date_quality_check")
        add("trend_analysis")
        add("chart_generation")
    elif _goal_has_any(lowered, ("region", "category", "segment", "group", "compare", "weakest", "strongest")):
        if "group_comparison" in by_tool:
            add("group_comparison")
            add("chart_generation")
    elif _goal_has_any(lowered, ("correlation", "relationship", "discount", "profit", "risk", "risky")):
        add("correlation_analysis")
        add("group_comparison")
    else:
        return []

    return selected


def _default_steps(candidates: list[PlanStep], scores: dict[str, float], profile: DatasetProfile) -> list[PlanStep]:
    by_tool = {step.tool_name: step for step in candidates}
    text_only = bool(profile.text_columns) and not profile.numeric_columns
    preferred_order = [
        "dataset_summary",
        "missing_value_check",
        "correlation_analysis",
        "group_comparison",
        "trend_analysis",
        "date_quality_check",
        "text_analysis",
        "topic_modeling",
    ]
    if not text_only:
        preferred_order.append("chart_generation")
    selected = [by_tool[tool] for tool in preferred_order if tool in by_tool]
    head = selected[:1]
    tail = sorted(selected[1:], key=lambda step: scores.get(step.tool_name, 1.0), reverse=True)
    return head + tail


def _goal_has_any(goal: str, tokens: tuple[str, ...]) -> bool:
    for token in tokens:
        if " " in token or "-" in token:
            if token in goal:
                return True
        elif re.search(rf"\b{re.escape(token)}\b", goal):
            return True
    return False


def _request_llm_plan(
    goal: str,
    profile: DatasetProfile,
    tool_scores: dict[str, float],
    llm_client: OpenAICompatibleClient,
) -> str:
    system = (
        "You are a data analysis planning agent. Select only from the allowed tools. "
        "Choose only the 2 to 4 tools that best match the user goal and observed dataset state. "
        "Do not select every tool by default. Do not write Python code. Return only JSON."
    )
    user = {
        "goal": goal,
        "dataset_profile": profile.to_dict(),
        "allowed_tools": ALLOWED_TOOL_DESCRIPTIONS,
        "recommended_applicable_tools": [step.tool_name for step in _candidate_steps(profile)],
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
        "planning_rules": [
            "Select dataset_summary when baseline statistics are useful.",
            "Prefer trend_analysis only for time/date goals.",
            "Prefer date_quality_check before trend_analysis when date-like columns exist.",
            "Use trend_analysis only when dataset_profile.date_columns is non-empty.",
            "Prefer text_analysis for text, feedback, comment, review, sentiment, keyword, theme, or topic goals.",
            "Prefer topic_modeling for theme, topic, clustering, repeated issue, or voice-of-customer goals.",
            "Prefer statistical_testing for significant, p-value, hypothesis, evidence, or group difference goals.",
            "Prefer predictive_modeling for predict, forecast, projection, estimate, or future goals.",
            "Prefer causal_risk_analysis for causal, impact, driver, why, or effect goals, but never claim causality from CSV alone.",
            "Prefer group_comparison for compare, region, category, segment, strongest, or weakest goals.",
            "Use group_comparison only when categorical and numeric columns are both available.",
            "Do not use group_comparison for ID-like or high-cardinality categorical columns.",
            "If dataset_profile.generic_column_names is true, keep planning conservative unless the user goal names a clear column intent.",
            "Prefer correlation_analysis for relationship, discount, profit, or risk goals.",
            "Use correlation_analysis only when at least two numeric columns are available.",
            "For numeric-only datasets, prefer summary, correlation, and visual distribution; do not invent categorical groups.",
            "For small datasets under 20 rows, treat statistical testing and predictive modeling as exploratory and select them only when the user explicitly asks.",
            "Prefer missing_value_check for quality, missing, audit, or reliability goals.",
            "Use missing_value_check only when missing values exist.",
            "For text-only datasets, prefer text_analysis and topic_modeling; do not use chart_generation to count raw full-sentence values.",
            "Do not request arbitrary code execution or unknown tools.",
        ],
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
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
            return json.dumps(payload)
        except json.JSONDecodeError:
            continue

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("LLM response did not contain JSON.")


def _deduplicate_steps(steps: list[PlanStep]) -> list[PlanStep]:
    seen: set[str] = set()
    result: list[PlanStep] = []
    for step in steps:
        if step.tool_name in seen:
            continue
        seen.add(step.tool_name)
        result.append(step)
    return result
