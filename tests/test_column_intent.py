import unittest

import pandas as pd

from agent.column_intent import describe_column_intent, select_group_columns, select_metric
from agent.perception import perceive_dataset


class ColumnIntentTest(unittest.TestCase):
    def test_selects_metric_named_in_goal(self):
        df = pd.DataFrame(
            {
                "region": ["North", "South"],
                "sales": [100, 120],
                "profit": [20, 35],
            }
        )
        profile = perceive_dataset(df)

        self.assertEqual(select_metric(profile, "Compare profit by region"), "profit")

    def test_selects_semantic_group_named_in_goal(self):
        df = pd.DataFrame(
            {
                "customer_segment": ["SMB", "Enterprise", "SMB"],
                "region": ["North", "South", "North"],
                "sales": [100, 120, 90],
            }
        )
        profile = perceive_dataset(df)

        groups = select_group_columns(df, profile, "Compare sales by customer segment")

        self.assertEqual(groups[0], "customer_segment")

    def test_describes_primary_column_intents(self):
        df = pd.DataFrame(
            {
                "order_date": ["2026-01-01", "2026-01-02"],
                "region": ["North", "South"],
                "revenue": [100, 120],
            }
        )
        profile = perceive_dataset(df)

        intent = describe_column_intent(df, profile, "Analyze revenue trend by region")

        self.assertEqual(intent["primary_metric"], "revenue")
        self.assertEqual(intent["primary_groups"][0], "region")
        self.assertEqual(intent["primary_date"], "order_date")


if __name__ == "__main__":
    unittest.main()
