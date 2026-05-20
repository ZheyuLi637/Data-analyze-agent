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


if __name__ == "__main__":
    unittest.main()
