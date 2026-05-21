import unittest

import pandas as pd

from agent.agent_core import DataAnalysisAgent, _synthesis_constraints
from agent.llm_client import LLMConfig, OpenAICompatibleClient
from agent.memory import initial_tool_scores, update_tool_scores


class AgentCoreTest(unittest.TestCase):
    def test_agent_returns_complete_trace(self):
        df = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "region": ["North", "South"],
                "sales": [100, 120],
                "profit": [20, 30],
            }
        )
        disabled_client = OpenAICompatibleClient(
            LLMConfig(enabled=False, api_key="", base_url="", model="")
        )

        run = DataAnalysisAgent(disabled_client).run(df, "find patterns", initial_tool_scores())
        stages = [item["stage"] for item in run.trace]

        self.assertIn("perceive", stages)
        self.assertIn("plan", stages)
        self.assertIn("act", stages)
        self.assertIn("observe", stages)
        self.assertIn("final_answer", stages)
        self.assertGreaterEqual(len(run.tools_used), 1)

    def test_feedback_updates_scores(self):
        scores = initial_tool_scores()
        updated = update_tool_scores(scores, ["dataset_summary"], "useful")

        self.assertGreater(updated["dataset_summary"], scores["dataset_summary"])

    def test_empty_dataset_returns_no_data_plan(self):
        df = pd.DataFrame(columns=["date", "sales"])
        disabled_client = OpenAICompatibleClient(
            LLMConfig(enabled=False, api_key="", base_url="", model="")
        )

        run = DataAnalysisAgent(disabled_client).run(df, "analyze this data", initial_tool_scores())

        self.assertEqual(run.plan.source, "no_data")
        self.assertEqual(run.tool_results, [])
        self.assertIn("Dataset has no rows", run.final_answer)

    def test_synthesis_constraints_warn_for_unreliable_dates(self):
        df = pd.DataFrame({"date": ["bad", "unknown"], "sales": [100, 120]})
        profile = DataAnalysisAgent(
            OpenAICompatibleClient(LLMConfig(enabled=False, api_key="", base_url="", model=""))
        ).run(df, "Analyze sales trends over time", initial_tool_scores()).profile

        constraints = _synthesis_constraints("Analyze sales trends over time", profile)

        self.assertTrue(any("date column was not reliably detected" in item for item in constraints))

    def test_uninterpretable_goal_executes_no_tools(self):
        df = pd.DataFrame({"sales": [100, 120], "profit": [20, 30]})
        disabled_client = OpenAICompatibleClient(
            LLMConfig(enabled=False, api_key="", base_url="", model="")
        )

        run = DataAnalysisAgent(disabled_client).run(df, "啦啦啦", initial_tool_scores())

        self.assertEqual(run.plan.source, "needs_clarification")
        self.assertEqual(run.tool_results, [])
        self.assertTrue(run.clarification["requires_user_input"])

    def test_broad_but_meaningful_goal_still_runs(self):
        df = pd.DataFrame({"sales": [100, 120], "profit": [20, 30]})
        disabled_client = OpenAICompatibleClient(
            LLMConfig(enabled=False, api_key="", base_url="", model="")
        )

        run = DataAnalysisAgent(disabled_client).run(df, "Analyze this data", initial_tool_scores())

        self.assertNotEqual(run.plan.source, "needs_clarification")
        self.assertGreater(len(run.tool_results), 0)

    def test_llm_failure_uses_polished_local_summary(self):
        df = pd.DataFrame({"region": ["North", "South"], "sales": [100, 120]})

        run = DataAnalysisAgent(FailingClient()).run(
            df,
            "Compare average sales across region",
            initial_tool_scores(),
        )

        self.assertIn("Summary:", run.final_answer)
        self.assertIn("Key findings:", run.final_answer)
        self.assertIn("LLM explanation was unavailable", run.final_answer)
        self.assertNotIn("Client Error", run.final_answer)
        self.assertNotIn("Traceback", run.final_answer)
        self.assertNotIn("LLM synthesis failed", run.final_answer)

    def test_trend_prompt_produces_trend_focused_summary(self):
        df = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "sales": [100, 120, 140],
            }
        )
        disabled_client = OpenAICompatibleClient(
            LLMConfig(enabled=False, api_key="", base_url="", model="")
        )

        run = DataAnalysisAgent(disabled_client).run(df, "Analyze sales trends over time", initial_tool_scores())

        self.assertIn("Analyzed sales over date", run.final_answer)
        self.assertIn("test time-based movement", run.final_answer)

    def test_group_prompt_mentions_strongest_group(self):
        df = pd.DataFrame({"region": ["North", "South", "North"], "sales": [100, 70, 120]})
        disabled_client = OpenAICompatibleClient(
            LLMConfig(enabled=False, api_key="", base_url="", model="")
        )

        run = DataAnalysisAgent(disabled_client).run(df, "Compare sales by region", initial_tool_scores())

        self.assertIn("top average groups", run.final_answer)
        self.assertIn("region: North", run.final_answer)

    def test_missing_values_summary_mentions_quality_risk(self):
        df = pd.DataFrame({"sales": [100, None], "profit": [20, None]})
        disabled_client = OpenAICompatibleClient(
            LLMConfig(enabled=False, api_key="", base_url="", model="")
        )

        run = DataAnalysisAgent(disabled_client).run(df, "Audit this dataset for data quality issues", initial_tool_scores())

        self.assertIn("data-quality risk", run.final_answer)
        self.assertIn("Missing values may affect reliability", run.final_answer)

    def test_messy_date_summary_warns_before_trend_claim(self):
        df = pd.DataFrame({"date": ["bad", "unknown"], "sales": [100, 120]})
        disabled_client = OpenAICompatibleClient(
            LLMConfig(enabled=False, api_key="", base_url="", model="")
        )

        run = DataAnalysisAgent(disabled_client).run(df, "Analyze sales trends over time", initial_tool_scores())

        self.assertIn("Clean the date column", run.final_answer)
        self.assertIn("date column was not reliably detected", run.final_answer)

    def test_followup_goal_uses_prior_context(self):
        df = pd.DataFrame({"region": ["North", "South"], "sales": [100, 120], "profit": [20, 30]})
        disabled_client = OpenAICompatibleClient(
            LLMConfig(enabled=False, api_key="", base_url="", model="")
        )

        run = DataAnalysisAgent(disabled_client).run(
            df,
            "continue with the same analysis",
            initial_tool_scores(),
            prior_context=[{"goal": "Compare sales by region", "final_answer": "North and South were compared."}],
        )

        self.assertIn("memory", [item["stage"] for item in run.trace])
        self.assertGreater(len(run.tool_results), 0)

    def test_followup_goal_keeps_previous_tools_in_memory_trace(self):
        df = pd.DataFrame({"region": ["North", "South"], "sales": [100, 120], "profit": [20, 30]})
        disabled_client = OpenAICompatibleClient(
            LLMConfig(enabled=False, api_key="", base_url="", model="")
        )

        run = DataAnalysisAgent(disabled_client).run(
            df,
            "compare that again with profit",
            initial_tool_scores(),
            prior_context=[
                {
                    "goal": "Compare sales by region",
                    "final_answer": "North and South were compared.",
                    "tools_used": ["group_comparison", "chart_generation"],
                }
            ],
        )

        memory_items = [item for item in run.trace if item["stage"] == "memory"]
        self.assertEqual(memory_items[0]["content"]["previous_tools"], ["group_comparison", "chart_generation"])
        self.assertIn("group_comparison", run.tools_used)

    def test_advanced_tools_are_summarized_without_raw_template(self):
        df = pd.DataFrame(
            {
                "week": [1, 2, 3, 4, 5],
                "sales": [100, 120, 140, 155, 170],
                "profit": [20, 24, 28, 30, 35],
            }
        )
        disabled_client = OpenAICompatibleClient(
            LLMConfig(enabled=False, api_key="", base_url="", model="")
        )

        run = DataAnalysisAgent(disabled_client).run(df, "Forecast future sales", initial_tool_scores())

        self.assertIn("Built a simple predictive model", run.final_answer)
        self.assertIn("simple local baseline", run.final_answer)


if __name__ == "__main__":
    unittest.main()


class FailingClient:
    @property
    def ready(self):
        return True

    def chat(self, messages, temperature=0.2):
        raise RuntimeError("429 Client Error: Too Many Requests")
