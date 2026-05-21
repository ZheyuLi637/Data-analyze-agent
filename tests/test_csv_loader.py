import unittest
from pathlib import Path

from agent.csv_loader import load_csv_bytes, load_csv_path


class CSVLoaderTest(unittest.TestCase):
    def test_loads_missing_value_fixture(self):
        loaded = load_csv_path(Path("data/edge_many_missing.csv"))

        self.assertIsNone(loaded.error)
        self.assertIsNotNone(loaded.dataframe)
        self.assertEqual(loaded.dataframe.shape[0], 8)

    def test_loads_header_only_fixture_as_zero_rows(self):
        loaded = load_csv_path(Path("data/edge_empty_header_only.csv"))

        self.assertIsNone(loaded.error)
        self.assertIsNotNone(loaded.dataframe)
        self.assertEqual(loaded.dataframe.shape, (0, 4))

    def test_loads_no_header_fixture_with_generated_column_names(self):
        loaded = load_csv_path(Path("data/edge_no_header.csv"), has_header=False)

        self.assertIsNone(loaded.error)
        self.assertIsNotNone(loaded.dataframe)
        self.assertEqual(list(loaded.dataframe.columns), ["column_1", "column_2", "column_3", "column_4"])
        self.assertEqual(loaded.dataframe.shape[0], 4)

    def test_reports_empty_file_error(self):
        loaded = load_csv_bytes(b"", "empty.csv")

        self.assertIsNotNone(loaded.error)
        self.assertIsNone(loaded.dataframe)


if __name__ == "__main__":
    unittest.main()
