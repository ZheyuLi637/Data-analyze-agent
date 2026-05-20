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


if __name__ == "__main__":
    unittest.main()

