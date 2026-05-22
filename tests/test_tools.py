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
        self.assertIn("bonferroni_p_value", result.table.columns)
        self.assertGreaterEqual(len(result.figure.axes), 2)

    def test_group_comparison_includes_multiple_categorical_columns(self):
        result = execute_tool("group_comparison", self.df, self.profile)

        self.assertIsNotNone(result.table)
        self.assertIn("group_column", result.table.columns)
        self.assertIn("region", set(result.table["group_column"]))
        self.assertIn("side-by-side bar charts", result.observation)
        self.assertGreaterEqual(len(result.figure.axes), 1)

    def test_group_comparison_uses_goal_matched_metric(self):
        result = execute_tool("group_comparison", self.df, self.profile, "Compare profit by region")

        self.assertIn("Compared profit by region", result.observation)

    def test_group_comparison_skips_high_cardinality_columns(self):
        df = pd.DataFrame({"user_id": [f"user_{index}" for index in range(80)], "sales": range(80)})
        profile = perceive_dataset(df)

        result = execute_tool("group_comparison", df, profile, "Compare sales by user")

        self.assertIn("Skipped because", result.observation)
        self.assertIsNone(result.figure)

    def test_trend_analysis_uses_multi_metric_visual(self):
        result = execute_tool("trend_analysis", self.df, self.profile)

        self.assertIsNotNone(result.figure)
        self.assertIn("multi-metric trend", result.observation)
        self.assertGreaterEqual(len(result.figure.axes[0].lines), 2)

    def test_trend_analysis_skips_unreliable_dates(self):
        df = pd.DataFrame({"date": ["2026-01-01", "bad", "unknown", "missing"], "sales": [1, 2, 3, 4]})
        profile = perceive_dataset(df)

        result = execute_tool("trend_analysis", df, profile, "Analyze sales trends over time")

        self.assertIn("Skipped because", result.observation)
        self.assertIsNone(result.figure)

    def test_date_quality_check_reports_parse_success(self):
        df = pd.DataFrame({"date": ["2026-01-01", "not-a-date", "2026/01/03"], "sales": [1, 2, 3]})
        profile = perceive_dataset(df)

        result = execute_tool("date_quality_check", df, profile)

        self.assertIn("parse success", result.observation)
        self.assertIn("parse_success_percent", result.table.columns)

    def test_text_analysis_extracts_keywords(self):
        df = pd.DataFrame(
            {
                "customer_feedback": [
                    "The onboarding was smooth and useful",
                    "The export was slow and confusing",
                    "Support solved the billing issue quickly",
                ]
            }
        )
        profile = perceive_dataset(df)

        result = execute_tool("text_analysis", df, profile, "Analyze feedback themes")

        self.assertIsNotNone(result.figure)
        self.assertIn("Top TF-IDF keywords", result.observation)
        self.assertIn("keyword", result.table.columns)
        self.assertIn("tfidf_score", result.table.columns)
        self.assertIn("document_count", result.table.columns)

    def test_topic_modeling_extracts_recurring_themes(self):
        df = pd.DataFrame(
            {
                "customer_feedback": [
                    "Export workflow is slow and export status is confusing",
                    "Export download failed and export retry is slow",
                    "Billing invoice is clear and billing support is useful",
                    "Billing support solved invoice question quickly",
                ]
            }
        )
        profile = perceive_dataset(df)

        result = execute_tool("topic_modeling", df, profile, "Find topic themes in feedback")

        self.assertIsNotNone(result.figure)
        self.assertIn("Modeled recurring topics", result.observation)
        self.assertIn("topic_label", result.table.columns)

    def test_statistical_testing_reports_p_value(self):
        df = pd.DataFrame(
            {
                "region": ["North", "North", "South", "South", "South"],
                "sales": [100, 110, 70, 72, 75],
            }
        )
        profile = perceive_dataset(df)

        result = execute_tool("statistical_testing", df, profile, "Test whether sales differs by region")

        self.assertIn("approximate p-value", result.observation)
        self.assertIn("approx_p_value", result.table.columns)

    def test_statistical_testing_uses_anova_for_three_groups(self):
        df = pd.DataFrame(
            {
                "region": ["North", "North", "South", "South", "West", "West"],
                "sales": [100, 110, 70, 72, 90, 95],
            }
        )
        profile = perceive_dataset(df)

        result = execute_tool("statistical_testing", df, profile, "Test whether sales differs by region")

        self.assertEqual(result.table.iloc[0]["test"], "One-way ANOVA")
        self.assertIn("approx_p_value", result.table.columns)

    def test_predictive_modeling_returns_baseline_fit(self):
        df = pd.DataFrame({"week": list(range(1, 11)), "sales": [10, 15, 21, 26, 30, 35, 40, 44, 49, 55]})
        profile = perceive_dataset(df)

        result = execute_tool("predictive_modeling", df, profile, "Forecast sales")

        self.assertIsNotNone(result.figure)
        self.assertIn("Built a simple predictive model", result.observation)
        self.assertIn("r_squared", result.table.columns)
        self.assertIn("train_rows", result.table.columns)
        self.assertIn("test_mae", result.table.columns)

    def test_numeric_string_columns_are_used_by_tools(self):
        df = pd.DataFrame({"week": ["1", "2", "3", "4"], "sales": ["$10", "$15", "$20", "$25"]})
        profile = perceive_dataset(df)

        result = execute_tool("dataset_summary", df, profile)

        self.assertIn("sales", set(result.table["column"]))

    def test_skip_messages_use_consistent_format(self):
        df = pd.DataFrame({"note": ["only text rows", "still only text"]})
        profile = perceive_dataset(df)

        result = execute_tool("correlation_analysis", df, profile)

        self.assertIn("Skipped because", result.observation)
        self.assertIn("Next step:", result.observation)
        self.assertNotIn("Traceback", result.observation)

    def test_causal_risk_analysis_adds_guardrail(self):
        df = pd.DataFrame(
            {
                "campaign": ["A", "A", "B", "B"],
                "sales": [100, 120, 80, 82],
                "discount": [5, 6, 15, 14],
            }
        )
        profile = perceive_dataset(df)

        result = execute_tool("causal_risk_analysis", df, profile, "Did campaign cause higher sales?")

        self.assertIn("does not claim causality", result.observation)
        self.assertIn("causal_guardrail", result.table.columns)

    def test_chart_generation_adds_distribution_and_boxplot(self):
        result = execute_tool("chart_generation", self.df, self.profile)

        self.assertIsNotNone(result.figure)
        self.assertIn("distribution and boxplot", result.observation)
        self.assertIn("Chart explanation", result.observation)
        self.assertEqual(len(result.figure.axes), 2)

    def test_missing_value_check_reports_severity_and_affected_rows(self):
        df = pd.DataFrame(
            {
                "region": ["North", "South", "North"],
                "sales": [100, None, None],
                "profit": [20, 30, None],
            }
        )
        profile = perceive_dataset(df)

        result = execute_tool("missing_value_check", df, profile, "Audit missing values by region")

        self.assertIn("rows", result.observation)
        self.assertIn("Highest missing column", result.observation)
        self.assertIn("severity", result.table.columns)

    def test_unknown_tool_raises(self):
        with self.assertRaises(ValueError):
            execute_tool("run_python", self.df, self.profile)


if __name__ == "__main__":
    unittest.main()
