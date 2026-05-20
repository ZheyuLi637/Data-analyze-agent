import unittest

from agent.evaluation import SCENARIOS, run_evaluation


class EvaluationDashboardTest(unittest.TestCase):
    def test_evaluation_suite_covers_all_scenarios(self):
        results = run_evaluation(use_llm=False)

        self.assertEqual(len(results), len(SCENARIOS))

    def test_evaluation_suite_passes_with_fallback_policy(self):
        results = run_evaluation(use_llm=False)
        failures = [result.to_dict() for result in results if not result.passed]

        self.assertEqual(failures, [])

    def test_safety_case_blocks_without_tools(self):
        results = {result.name: result for result in run_evaluation(use_llm=False)}

        safety = results["Safety guardrail"]
        self.assertTrue(safety.passed)
        self.assertEqual(safety.plan_source, "blocked")
        self.assertEqual(safety.tools, [])

    def test_header_only_case_reports_no_data(self):
        results = {result.name: result for result in run_evaluation(use_llm=False)}

        header_only = results["Header-only CSV"]
        self.assertTrue(header_only.passed)
        self.assertEqual(header_only.plan_source, "no_data")
        self.assertEqual(header_only.tools, [])


if __name__ == "__main__":
    unittest.main()
