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

    def test_auto_detects_header_after_preamble(self):
        loaded = load_csv_path(Path("data/edge_complex_header.csv"), has_header="auto")

        self.assertIsNone(loaded.error)
        self.assertIsNotNone(loaded.dataframe)
        self.assertEqual(loaded.detected_header_row, 2)
        self.assertEqual(list(loaded.dataframe.columns), ["date", "region", "sales", "profit"])
        self.assertEqual(loaded.dataframe.shape[0], 4)

    def test_limits_large_csv_rows(self):
        content = "date,sales\n" + "\n".join(f"2026-01-{(index % 28) + 1:02d},{index}" for index in range(150))

        loaded = load_csv_bytes(content.encode("utf-8"), "large.csv", max_rows=25)

        self.assertIsNone(loaded.error)
        self.assertTrue(loaded.sampled)
        self.assertEqual(loaded.original_row_count, 150)
        self.assertEqual(loaded.dataframe.shape[0], 25)

    def test_reports_empty_file_error(self):
        loaded = load_csv_bytes(b"", "empty.csv")

        self.assertIsNotNone(loaded.error)
        self.assertIsNone(loaded.dataframe)


if __name__ == "__main__":
    unittest.main()
