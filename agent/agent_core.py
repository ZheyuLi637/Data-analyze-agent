from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agent.clarification import clarification_context
from agent.column_intent import describe_column_intent
from agent.guardrails import evaluate_guardrail, guardrail_message
from agent.llm_client import OpenAICompatibleClient, compact_json
from agent.memory import initial_tool_scores
from agent.perception import DatasetProfile, perceive_dataset
from agent.planner import PlanResult, plan_analysis
from agent.tools import ToolResult, execute_tool


@dataclass
class AgentRun:
    goal: str
    profile: DatasetProfile
    plan: PlanResult
    tool_results: list[ToolResult]
    final_answer: str
    trace: list[dict]
    clarification: dict
    guardrail: dict

    @property
    def tools_used(self) -> list[str]:
        return [result.name for result in self.tool_results]


class DataAnalysisAgent:
    def __init__(self, llm_client: OpenAICompatibleClient | None = None) -> None:
        self.llm_client = llm_client or OpenAICompatibleClient()

    def run(
        self,
        df: pd.DataFrame,
        goal: str,
        tool_scores: dict[str, float] | None = None,
        prior_context: list[dict] | None = None,
    ) -> AgentRun:
        scores = tool_scores or initial_tool_scores()
        trace: list[dict] = [{"stage": "goal", "content": goal}]
        memory_context = _memory_context(prior_context or [])
        effective_goal = _goal_with_memory(goal, memory_context)
        if effective_goal != goal:
            trace.append({"stage": "memory", "content": memory_context})

        profile = perceive_dataset(df)
        trace.append({"stage": "perceive", "content": profile.to_dict()})
        column_intent = describe_column_intent(df, profile, effective_goal)
        trace.append({"stage": "column_intent", "content": column_intent})

        if profile.row_count == 0:
            plan = PlanResult(steps=[], source="no_data", error="Dataset has no rows.")
            final_answer = "Dataset has no rows. Upload a CSV with at least one data row before running analysis."
            trace.append({"stage": "plan", "content": plan.to_dict()})
            trace.append({"stage": "final_answer", "content": final_answer})
            return AgentRun(
                goal=goal,
                profile=profile,
                plan=plan,
                tool_results=[],
                final_answer=final_answer,
                trace=trace,
                clarification={"ambiguous": False, "suggestions": [], "planning_goal": goal},
                guardrail={"blocked": False, "reason": "", "matched_terms": []},
            )

        guardrail = evaluate_guardrail(goal)
        trace.append({"stage": "guardrail", "content": guardrail})
        if guardrail["blocked"]:
            plan = PlanResult(steps=[], source="blocked", error=guardrail["reason"])
            final_answer = guardrail_message(guardrail)
            trace.append({"stage": "plan", "content": plan.to_dict()})
            trace.append({"stage": "final_answer", "content": final_answer})
            return AgentRun(
                goal=goal,
                profile=profile,
                plan=plan,
                tool_results=[],
                final_answer=final_answer,
                trace=trace,
                clarification={"ambiguous": False, "suggestions": [], "planning_goal": goal},
                guardrail=guardrail,
            )

        clarification = clarification_context(effective_goal, profile)
        trace.append({"stage": "clarify", "content": clarification})
        if clarification["requires_user_input"]:
            plan = PlanResult(steps=[], source="needs_clarification", error="Goal is not interpretable enough to choose safe actions.")
            final_answer = (
                "I could not identify a data-analysis goal from this input. "
                "Choose one of the clarification suggestions or describe what pattern, risk, trend, relationship, or data-quality issue you want analyzed."
            )
            trace.append({"stage": "plan", "content": plan.to_dict()})
            trace.append({"stage": "final_answer", "content": final_answer})
            return AgentRun(
                goal=goal,
                profile=profile,
                plan=plan,
                tool_results=[],
                final_answer=final_answer,
                trace=trace,
                clarification=clarification,
                guardrail=guardrail,
            )

        plan = plan_analysis(clarification["planning_goal"], profile, scores, self.llm_client)
        trace.append({"stage": "plan", "content": plan.to_dict()})

        tool_results: list[ToolResult] = []
        for step in plan.steps:
            result = execute_tool(step.tool_name, df, profile, clarification["planning_goal"])
            tool_results.append(result)
            trace.append(
                {
                    "stage": "act",
                    "tool": result.name,
                    "reason": step.reason,
                    "observation": result.observation,
                }
            )

        final_answer = self._synthesize(clarification["planning_goal"], profile, plan, tool_results)
        trace.append({"stage": "observe", "content": [result.observation for result in tool_results]})
        trace.append({"stage": "final_answer", "content": final_answer})

        return AgentRun(
            goal=goal,
            profile=profile,
            plan=plan,
            tool_results=tool_results,
            final_answer=final_answer,
            trace=trace,
            clarification=clarification,
            guardrail=guardrail,
        )

    def _synthesize(
        self,
        goal: str,
        profile: DatasetProfile,
        plan: PlanResult,
        tool_results: list[ToolResult],
    ) -> str:
        observations = [result.observation for result in tool_results]
        if self.llm_client.ready:
            try:
                return self._llm_synthesis(goal, profile, plan, observations)
            except Exception:
                return _grounded_summary(goal, profile, plan, tool_results, llm_unavailable=True)
        return _grounded_summary(goal, profile, plan, tool_results)

    def _llm_synthesis(
        self,
        goal: str,
        profile: DatasetProfile,
        plan: PlanResult,
        observations: list[str],
    ) -> str:
        system = (
            "You are a data analysis agent. Summarize grounded findings from tool observations. "
            "Do not invent statistics that are not present in the observations."
        )
        user = {
            "goal": goal,
            "dataset_profile": profile.to_dict(),
            "plan_source": plan.source,
            "plan_error": plan.error,
            "tool_observations": observations,
            "constraints": _synthesis_constraints(goal, profile),
            "format": "Return 4 concise bullets: state observed, actions taken, key findings, next step.",
        }
        return self.llm_client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": compact_json(user)},
            ],
            temperature=0.2,
        )


def _grounded_summary(
    goal: str,
    profile: DatasetProfile,
    plan: PlanResult,
    tool_results: list[ToolResult],
    llm_unavailable: bool = False,
) -> str:
    observations = [result.observation for result in tool_results]
    findings = _key_findings(observations)
    if not findings:
        findings = ["The agent did not find enough grounded tool output to make an analytical claim."]

    action_reason = _action_reason(plan, tool_results)
    next_step = _next_step(goal, profile, observations)
    caveats = _caveats(goal, profile, observations, llm_unavailable)

    lines = [
        f"Summary: I analyzed {profile.row_count} rows and {profile.column_count} columns for this goal: {goal}",
        "",
        "Key findings:",
    ]
    lines.extend(f"- {finding}" for finding in findings[:4])
    lines.extend(["", f"Action reasoning: {action_reason}", f"Next step: {next_step}"])
    if caveats:
        lines.extend(["", "Caveat: " + " ".join(caveats)])
    return "\n".join(lines)


def _key_findings(observations: list[str]) -> list[str]:
    findings: list[str] = []
    for observation in observations:
        if observation.startswith("Generated summary statistics"):
            findings.append(observation.replace("Generated", "The agent generated"))
        elif observation.startswith("No numeric columns"):
            findings.append("The dataset is text/categorical only, so numeric statistics and correlations were skipped.")
        elif observation.startswith("Found "):
            findings.append(observation + " Treat this as a data-quality risk before trusting downstream patterns.")
        elif observation.startswith("Computed correlations"):
            findings.append(observation)
        elif observation.startswith("Compared "):
            findings.append(observation)
        elif observation.startswith("Analyzed "):
            findings.append(observation)
        elif observation.startswith("Modeled recurring topics"):
            findings.append(observation)
        elif observation.startswith("Ran statistical testing"):
            findings.append(observation)
        elif observation.startswith("Built a simple predictive model"):
            findings.append(observation)
        elif observation.startswith("Audited causal risk"):
            findings.append(observation)
        elif observation.startswith("Checked date quality"):
            findings.append(observation)
        elif observation.startswith("Generated distribution") or observation.startswith("Generated count"):
            findings.append(observation)
        elif observation.startswith("Skipped "):
            findings.append(observation)
    return findings


def _action_reason(plan: PlanResult, tool_results: list[ToolResult]) -> str:
    tools = [result.name for result in tool_results]
    if not tools:
        return "No analysis tools were executed because the agent could not safely proceed."

    reasons = {
        "dataset_summary": "establish baseline statistics",
        "missing_value_check": "check reliability before interpretation",
        "correlation_analysis": "measure numeric relationships",
        "group_comparison": "compare performance across categories",
        "trend_analysis": "test time-based movement",
        "date_quality_check": "verify date parsing quality",
        "text_analysis": "extract text themes and sentiment",
        "topic_modeling": "group recurring text topics",
        "statistical_testing": "check whether patterns have statistical support",
        "predictive_modeling": "fit a simple baseline forecast",
        "causal_risk_analysis": "separate causal hypotheses from observational evidence",
        "chart_generation": "provide visual support",
    }
    selected = [reasons.get(tool, tool.replace("_", " ")) for tool in tools]
    source = "LLM planner" if plan.source == "llm" else "validated fallback policy"
    return f"The {source} selected these actions to " + ", ".join(selected) + "."


def _next_step(goal: str, profile: DatasetProfile, observations: list[str]) -> str:
    lowered = goal.lower()
    if any("Found " in observation and "missing values" in observation for observation in observations):
        return "Clean or explain the missing values, then rerun the same analysis to check whether the findings change."
    if any(token in lowered for token in ("trend", "over time", "date", "time")) and not profile.date_columns:
        return "Clean the date column into a consistent datetime format before making any trend claim."
    if any("Date cleanup is recommended" in observation for observation in observations):
        return "Standardize invalid date values, then rerun trend analysis to compare cleaned and original results."
    if any(observation.startswith("Analyzed text column") for observation in observations):
        return "Review the top keywords and sentiment split, then group the original text rows into recurring issue themes."
    if any(observation.startswith("Modeled recurring topics") for observation in observations):
        return "Inspect the example rows behind each topic and decide which theme should drive the next product or operations action."
    if any(observation.startswith("Ran statistical testing") for observation in observations):
        return "Validate the tested pattern with more rows or a holdout slice before using it as decision evidence."
    if any(observation.startswith("Built a simple predictive model") for observation in observations):
        return "Compare this baseline prediction with a holdout period before relying on it for planning."
    if any(observation.startswith("Audited causal risk") for observation in observations):
        return "Define a controlled comparison or add confounder columns before making a causal claim."
    if any("Top pairs:" in observation for observation in observations):
        return "Inspect the strongest numeric relationships in the correlation table and validate whether they match domain expectations."
    if any("top average groups" in observation for observation in observations):
        return "Review the highest and lowest groups in the group comparison table and investigate what drives the gap."
    if not profile.numeric_columns:
        return "Use a text-specific method such as sentiment or topic analysis if deeper feedback interpretation is required."
    return "Review the generated outputs, then give feedback so the agent can prioritize better tools in the next run."


def _caveats(
    goal: str,
    profile: DatasetProfile,
    observations: list[str],
    llm_unavailable: bool,
) -> list[str]:
    caveats: list[str] = []
    if llm_unavailable:
        caveats.append("LLM explanation was unavailable, so this response uses the local grounded summarizer.")
    if any(count > 0 for count in profile.missing_values.values()):
        caveats.append("Missing values may affect reliability.")
    if 0 < profile.row_count < 8:
        caveats.append(f"Only {profile.row_count} rows were analyzed, so findings should be treated as directional.")
    if any(token in goal.lower() for token in ("trend", "over time", "date", "time")) and not profile.date_columns:
        caveats.append("The date column was not reliably detected, so trend analysis should wait until date cleanup.")
    weak_date_columns = [
        column
        for column, success_percent in profile.date_parse_percent.items()
        if success_percent < 80
    ]
    if weak_date_columns:
        caveats.append(f"Date cleanup is recommended for: {', '.join(weak_date_columns)}.")
    if not profile.numeric_columns:
        caveats.append("No numeric columns were detected.")
    if profile.generic_column_names:
        caveats.append("Column names look auto-generated, so the agent keeps interpretation broad until headers are clarified.")
    if profile.inferred_numeric_columns:
        caveats.append(f"Numeric-looking strings were converted for analysis: {', '.join(profile.inferred_numeric_columns)}.")
    high_cardinality_columns = _high_cardinality_columns(profile)
    if high_cardinality_columns:
        caveats.append(
            "High-cardinality categorical columns were treated as identifiers rather than grouping fields: "
            + ", ".join(high_cardinality_columns)
            + "."
        )
    if any(observation.startswith("Audited causal risk") for observation in observations):
        caveats.append("Causal findings are hypothesis-generating only because the data is observational.")
    if any(observation.startswith("Built a simple predictive model") for observation in observations):
        caveats.append("The prediction is a simple local baseline, not a validated production forecast.")
        if profile.row_count < 20:
            caveats.append("The sample is small, so model metrics are demo-level and may overfit.")
    if any(observation.startswith("Ran statistical testing") and "approximate p-value" in observation for observation in observations):
        caveats.append("Statistical p-values use a lightweight normal approximation because the prototype avoids heavy dependencies.")
        if profile.row_count < 20:
            caveats.append("Small samples and highly related variables can make statistical signals look stronger than they are.")
    if any(observation.startswith("Skipped ") for observation in observations):
        caveats.append("At least one requested analysis was skipped because the dataset did not support it.")
    return caveats


def _high_cardinality_columns(profile: DatasetProfile) -> list[str]:
    row_count = max(profile.row_count, 1)
    threshold = min(30, max(2, int(row_count * 0.5)))
    return [
        column
        for column, cardinality in profile.categorical_cardinality.items()
        if cardinality > threshold
    ]


def _memory_context(prior_context: list[dict]) -> dict:
    if not prior_context:
        return {}
    last = prior_context[-1]
    return {
        "previous_goal": str(last.get("goal", ""))[:220],
        "previous_answer": str(last.get("final_answer", ""))[:700].replace("\n", " "),
        "previous_tools": list(last.get("tools_used", last.get("tools", [])))[:8],
    }


def _goal_with_memory(goal: str, memory_context: dict) -> str:
    if not memory_context or not _looks_like_followup(goal):
        return goal
    previous_goal = memory_context.get("previous_goal", "")
    previous_answer = memory_context.get("previous_answer", "")
    previous_tools = ", ".join(memory_context.get("previous_tools", [])) or "none"
    return (
        f"{goal}\n"
        f"Follow-up context from previous agent run: previous goal was '{previous_goal}'. "
        f"Previous tools used: {previous_tools}. Previous answer summary: {previous_answer}. "
        "Resolve pronouns such as it, that, same, those, previous, and continue against this context before planning."
    )


def _looks_like_followup(goal: str) -> bool:
    lowered = goal.strip().lower()
    phrase_tokens = ("follow up", "what about", "compare that")
    word_tokens = ("continue", "also", "next", "same", "previous", "again", "it", "that")
    chinese_tokens = ("继续", "接着", "再分析", "那", "还有", "一样", "上一个", "刚才")
    if any(token in lowered for token in phrase_tokens + chinese_tokens):
        return True
    import re

    return any(re.search(rf"\b{re.escape(token)}\b", lowered) for token in word_tokens)


def _synthesis_constraints(goal: str, profile: DatasetProfile) -> list[str]:
    constraints: list[str] = []
    lowered = goal.lower()
    if any(token in lowered for token in ("trend", "over time", "date", "time")) and not profile.date_columns:
        constraints.append(
            "The date column was not reliably detected. Do not claim trend analysis is possible until dates are cleaned."
        )
    if not profile.numeric_columns:
        constraints.append("No numeric columns were detected. Do not claim numeric statistics or correlations.")
    if not profile.categorical_columns:
        constraints.append("No categorical columns were detected. Do not claim group comparison results.")
    if any(token in lowered for token in ("causal", "cause", "impact", "effect", "driver")):
        constraints.append("This prototype can audit causal risk but must not claim causality from observational CSV data alone.")
    if any(token in lowered for token in ("predict", "forecast", "projection", "future")):
        constraints.append("Prediction output is a simple local baseline and needs holdout validation before operational use.")
    return constraints
