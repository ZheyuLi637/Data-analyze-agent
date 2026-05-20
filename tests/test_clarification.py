import unittest

import pandas as pd

from agent.clarification import (
    clarification_context,
    goal_has_analysis_intent,
    goal_is_ambiguous,
    suggest_clarifications,
)
from agent.perception import perceive_dataset


class ClarificationTest(unittest.TestCase):
    def test_detects_ambiguous_goal(self):
        self.assertTrue(goal_is_ambiguous("analyze this data"))
        self.assertTrue(goal_is_ambiguous("看看这个数据有什么问题"))

    def test_accepts_specific_goal(self):
        self.assertFalse(goal_is_ambiguous("Compare sales by region and identify profit risks"))

    def test_suggests_dataset_specific_focus(self):
        df = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "region": ["North", "South"],
                "sales": [100, 120],
                "profit": [20, 30],
            }
        )
        profile = perceive_dataset(df)

        suggestions = suggest_clarifications(profile)

        self.assertGreaterEqual(len(suggestions), 2)
        self.assertIn("sales", suggestions[0])

    def test_context_adds_planning_goal_for_ambiguous_prompt(self):
        df = pd.DataFrame({"sales": [100, 120], "profit": [20, 30]})
        profile = perceive_dataset(df)

        context = clarification_context("help me analyze", profile)

        self.assertTrue(context["ambiguous"])
        self.assertFalse(context["requires_user_input"])
        self.assertIn("Clarification suggestion used for planning", context["planning_goal"])

    def test_missing_values_are_suggested_first(self):
        df = pd.DataFrame({"date": ["2026-01-01", None], "sales": [100, None]})
        profile = perceive_dataset(df)

        suggestions = suggest_clarifications(profile)

        self.assertTrue(suggestions[0].startswith("Audit missing values"))

    def test_uninterpretable_goal_requires_user_input(self):
        df = pd.DataFrame({"sales": [100, 120], "profit": [20, 30]})
        profile = perceive_dataset(df)

        context = clarification_context("啦啦啦", profile)

        self.assertTrue(context["ambiguous"])
        self.assertFalse(goal_has_analysis_intent("啦啦啦", profile))
        self.assertTrue(context["requires_user_input"])
        self.assertNotIn("Clarification suggestion used for planning", context["planning_goal"])


if __name__ == "__main__":
    unittest.main()
