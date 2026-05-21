import unittest

import pandas as pd

from agent.perception import perceive_dataset
from agent.planner import fallback_plan, filter_applicable_steps, parse_llm_plan, plan_analysis


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

    def test_parse_json_with_trailing_text(self):
        raw = (
            '{"tools":[{"tool_name":"chart_generation","reason":"visual",'
            '"expected_insight":"chart"}]} trailing explanation'
        )

        steps = parse_llm_plan(raw)

        self.assertEqual([step.tool_name for step in steps], ["chart_generation"])

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
        self.assertLessEqual(len(tools), 4)

    def test_fallback_uses_goal_focus(self):
        df = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "region": ["North", "South"],
                "sales": [100, 120],
                "profit": [20, 30],
            }
        )
        profile = perceive_dataset(df)

        trend_tools = [step.tool_name for step in fallback_plan(profile, goal="Analyze sales trends over time")]
        group_tools = [step.tool_name for step in fallback_plan(profile, goal="Compare sales by region")]

        self.assertIn("trend_analysis", trend_tools)
        self.assertNotIn("group_comparison", trend_tools)
        self.assertIn("group_comparison", group_tools)
        self.assertNotIn("trend_analysis", group_tools)

    def test_plan_falls_back_without_llm(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        profile = perceive_dataset(df)

        result = plan_analysis("find patterns", profile, {})

        self.assertEqual(result.source, "fallback")
        self.assertGreaterEqual(len(result.steps), 1)

    def test_filters_inapplicable_llm_steps(self):
        df = pd.DataFrame({"date": ["bad", "unknown"], "sales": [100, 120]})
        profile = perceive_dataset(df)
        steps = parse_llm_plan(
            '{"tools":[{"tool_name":"trend_analysis","reason":"time","expected_insight":"trend"},'
            '{"tool_name":"chart_generation","reason":"visual","expected_insight":"chart"}]}'
        )

        filtered = filter_applicable_steps(steps, profile)

        self.assertEqual([step.tool_name for step in filtered], ["chart_generation"])

    def test_fallback_uses_text_focus(self):
        df = pd.DataFrame({"feedback": ["The product is useful but exports are slow."]})
        profile = perceive_dataset(df)

        tools = [step.tool_name for step in fallback_plan(profile, goal="Analyze feedback sentiment and themes")]

        self.assertIn("text_analysis", tools)

    def test_fallback_checks_date_quality_for_messy_dates(self):
        df = pd.DataFrame({"date": ["2026-01-01", "not-a-date", "unknown"], "sales": [100, 120, 90]})
        profile = perceive_dataset(df)

        tools = [step.tool_name for step in fallback_plan(profile, goal="Analyze sales trends over time")]

        self.assertIn("date_quality_check", tools)


if __name__ == "__main__":
    unittest.main()
