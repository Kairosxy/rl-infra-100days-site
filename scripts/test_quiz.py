"""Validate interview-bank data, number alignment and publication boundaries."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_quiz import TOTAL_QUESTIONS, generate_quiz, get_question_bank

ROOT = Path(__file__).resolve().parent.parent


class InterviewQuizTest(unittest.TestCase):
    def test_all_224_have_two_distinct_reference_explanations(self):
        data = get_question_bank(ROOT)
        self.assertEqual(data["question_count"], TOTAL_QUESTIONS)
        for index, item in enumerate(data["questions"], start=1):
            self.assertEqual(item["code"], f"Q{index:03d}")
            self.assertTrue(item["brief"].strip())
            self.assertTrue(item["deep"].strip())
            self.assertTrue(item["followup"].strip())
            self.assertIn(item["level"], {"基础", "进阶"})

    def test_standalone_page_has_all_questions_and_no_private_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quiz" / "index.html"
            self.assertEqual(generate_quiz(ROOT, output), TOTAL_QUESTIONS)
            content = output.read_text(encoding="utf-8")
            self.assertIn("Q001", content)
            self.assertIn("Q224", content)
            self.assertIn("我的回答", content)
            self.assertIn("导出练习记录", content)
            self.assertNotIn("{{QUIZ_DATA}}", content)
            self.assertNotIn("{{QUIZ_STYLE}}", content)
            self.assertNotIn("{{QUIZ_SCRIPT}}", content)
            self.assertNotIn("portfolio/project_narrative.md", content)
            self.assertNotIn("github.com/Kairosxy/rl-infra-100days/issues", content)

    def test_difficulty_and_course_link_exist(self):
        questions = get_question_bank(ROOT)["questions"]
        self.assertEqual(len([x for x in questions if x["level"] == "基础"]), 104)
        self.assertEqual(len([x for x in questions if x["level"] == "进阶"]), 120)
        self.assertTrue(all(1 <= x["course_day"] <= 100 for x in questions))


if __name__ == "__main__":
    unittest.main()
