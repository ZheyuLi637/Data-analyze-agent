import unittest

import pandas as pd

from agent.perception import perceive_dataset
from agent.planner import fallback_plan, parse_llm_plan, plan_analysis


class PlannerTest(unittest.TestCase):
    def test_parse_valid_llm_plan(self):
        raw = """
        {
          "tools": [
            {
              "tool_name": "dataset_summary",
              "reason": "Understand numeric fields",
              "expected_insight": "Baseline stats"
            }
          ]
        }
        """

        steps = parse_llm_plan(raw)

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].tool_name, "dataset_summary")

    def test_rejects_unknown_tool(self):
        raw = '{"tools":[{"tool_name":"run_python","reason":"bad","expected_insight":"bad"}]}'

        with self.assertRaises(ValueError):
            parse_llm_plan(raw)

    def test_fallback_selects_expected_tools(self):
        df = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "region": ["North", "South"],
                "sales": [100, 120],
                "profit": [20, 30],
            }
        )
        profile = perceive_dataset(df)

        tools = [step.tool_name for step in fallback_plan(profile)]

        self.assertIn("dataset_summary", tools)
        self.assertIn("correlation_analysis", tools)
        self.assertIn("group_comparison", tools)
        self.assertIn("trend_analysis", tools)
        self.assertIn("chart_generation", tools)

    def test_plan_falls_back_without_llm(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        profile = perceive_dataset(df)

        result = plan_analysis("find patterns", profile, {})

        self.assertEqual(result.source, "fallback")
        self.assertGreaterEqual(len(result.steps), 1)


if __name__ == "__main__":
    unittest.main()

