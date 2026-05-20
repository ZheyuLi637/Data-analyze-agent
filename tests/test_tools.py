import unittest

import pandas as pd

from agent.perception import perceive_dataset
from agent.tools import execute_tool


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "region": ["North", "South", "North"],
                "sales": [100, 120, 130],
                "profit": [20, 30, 35],
            }
        )
        self.profile = perceive_dataset(self.df)

    def test_summary_returns_table(self):
        result = execute_tool("dataset_summary", self.df, self.profile)

        self.assertIsNotNone(result.table)
        self.assertEqual(result.name, "dataset_summary")

    def test_correlation_returns_figure(self):
        result = execute_tool("correlation_analysis", self.df, self.profile)

        self.assertIsNotNone(result.figure)
        self.assertIsNotNone(result.table)
        self.assertIn("Top pairs", result.observation)
        self.assertIn("heatmap", result.observation)
        self.assertGreaterEqual(len(result.figure.axes), 2)

    def test_group_comparison_includes_multiple_categorical_columns(self):
        result = execute_tool("group_comparison", self.df, self.profile)

        self.assertIsNotNone(result.table)
        self.assertIn("group_column", result.table.columns)
        self.assertIn("region", set(result.table["group_column"]))
        self.assertIn("side-by-side bar charts", result.observation)
        self.assertGreaterEqual(len(result.figure.axes), 1)

    def test_trend_analysis_uses_multi_metric_visual(self):
        result = execute_tool("trend_analysis", self.df, self.profile)

        self.assertIsNotNone(result.figure)
        self.assertIn("multi-metric trend", result.observation)
        self.assertGreaterEqual(len(result.figure.axes[0].lines), 2)

    def test_chart_generation_adds_distribution_and_boxplot(self):
        result = execute_tool("chart_generation", self.df, self.profile)

        self.assertIsNotNone(result.figure)
        self.assertIn("distribution and boxplot", result.observation)
        self.assertEqual(len(result.figure.axes), 2)

    def test_unknown_tool_raises(self):
        with self.assertRaises(ValueError):
            execute_tool("run_python", self.df, self.profile)


if __name__ == "__main__":
    unittest.main()
