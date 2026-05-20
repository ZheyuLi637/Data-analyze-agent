import unittest

import pandas as pd

from agent.agent_core import DataAnalysisAgent
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


if __name__ == "__main__":
    unittest.main()
