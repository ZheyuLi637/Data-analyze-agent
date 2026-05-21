import unittest

import pandas as pd

from agent.perception import perceive_dataset


class PerceptionTest(unittest.TestCase):
    def test_detects_core_dataset_state(self):
        df = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "region": ["North", "South"],
                "sales": [100, None],
            }
        )

        profile = perceive_dataset(df)

        self.assertEqual(profile.row_count, 2)
        self.assertIn("sales", profile.numeric_columns)
        self.assertIn("region", profile.categorical_columns)
        self.assertIn("date", profile.date_columns)
        self.assertEqual(profile.missing_values["sales"], 1)

    def test_records_date_parse_quality_for_messy_dates(self):
        df = pd.DataFrame({"date": ["2026-01-01", "bad-date", "2026/01/03"], "sales": [1, 2, 3]})

        profile = perceive_dataset(df)

        self.assertIn("date", profile.date_parse_percent)
        self.assertGreater(profile.date_parse_percent["date"], 0)
        self.assertIn("bad-date", profile.date_invalid_examples["date"])


if __name__ == "__main__":
    unittest.main()
