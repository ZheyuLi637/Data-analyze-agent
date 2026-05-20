from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agent.clarification import clarification_context
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

        plan = plan_analysis(clarification["planning_goal"], profile, scores, self.llm_client)
        trace.append({"stage": "plan", "content": plan.to_dict()})

        tool_results: list[ToolResult] = []
        for step in plan.steps:
            result = execute_tool(step.tool_name, df, profile)
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
            except Exception as exc:
                return _fallback_summary(goal, profile, observations, f"LLM synthesis failed: {exc}")
        return _fallback_summary(goal, profile, observations)

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
            "tool_observations": observations,
            "format": "Return 4 concise bullets: state observed, actions taken, key findings, next step.",
        }
        return self.llm_client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": compact_json(user)},
            ],
            temperature=0.2,
        )


def _fallback_summary(
    goal: str,
    profile: DatasetProfile,
    observations: list[str],
    note: str | None = None,
) -> str:
    lines = [
        f"Goal: {goal}",
        f"Observed dataset state: {profile.row_count} rows, {profile.column_count} columns.",
        "Actions completed: " + "; ".join(observations),
    ]
    if note:
        lines.append(note)
    lines.append(
        "Next step: review the generated tables and charts, then provide feedback so the agent can adjust tool priorities."
    )
    return "\n".join(lines)
