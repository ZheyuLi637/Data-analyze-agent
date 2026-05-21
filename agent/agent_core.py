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
    ) -> AgentRun:
        scores = tool_scores or initial_tool_scores()
        trace: list[dict] = [{"stage": "goal", "content": goal}]

        profile = perceive_dataset(df)
        trace.append({"stage": "perceive", "content": profile.to_dict()})
        column_intent = describe_column_intent(df, profile, goal)
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

        clarification = clarification_context(goal, profile)
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

        final_answer = self._synthesize(goal, profile, plan, tool_results)
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
    if any("Top pairs:" in observation for observation in observations):
        return "Inspect the strongest numeric relationships in the correlation table and validate whether they match domain expectations."
    if any("top average groups" in observation for observation in observations):
        return "Review the highest and lowest groups in the group comparison table and investigate what drives the gap."
    if not profile.numeric_columns:
        return "Use a text-specific method such as sentiment or topic analysis if deeper feedback interpretation is required."
    return "Review the generated table and chart, then give feedback so the agent can prioritize better tools in the next run."


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
    if any(token in goal.lower() for token in ("trend", "over time", "date", "time")) and not profile.date_columns:
        caveats.append("The date column was not reliably detected, so trend analysis should wait until date cleanup.")
    if not profile.numeric_columns:
        caveats.append("No numeric columns were detected.")
    if any(observation.startswith("Skipped ") for observation in observations):
        caveats.append("At least one requested analysis was skipped because the dataset did not support it.")
    return caveats


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
    return constraints
