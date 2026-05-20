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

    def test_unknown_tool_raises(self):
        with self.assertRaises(ValueError):
            execute_tool("run_python", self.df, self.profile)


if __name__ == "__main__":
    unittest.main()

