"""Tests for the published interview question bank and privacy boundary."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from build_quiz import build, read_bank

ROOT = Path(__file__).resolve().parent.parent


class QuestionBankTest(unittest.TestCase):
    def test_all_questions_have_sources_and_references(self):
        data = read_bank(ROOT)
        self.assertEqual(data["question_count"], 224)
        self.assertEqual(len(data["questions"]), 224)
        self.assertEqual([q["id"] for q in data["questions"]], list(range(1, 225)))
        self.assertEqual(sum(q["level"] == "基础" for q in data["questions"]), 104)
        self.assertEqual(sum(q["level"] == "进阶" for q in data["questions"]), 120)
        self.assertTrue(all(q["brief"] and q["deep"] and q["followup"] for q in data["questions"]))

    def test_html_is_self_contained_and_contains_no_personal_answers(self):
        with tempfile.TemporaryDirectory() as temp:
            result = Path(temp) / "quiz" / "index.html"
            self.assertEqual(build(ROOT, result), 224)
            content = result.read_text(encoding="utf-8")
            self.assertIn("Q001", content)
            self.assertIn("Q224", content)
            self.assertIn("我的回答", content)
            self.assertIn("随机 10 题", content)
            self.assertIn("导出练习记录", content)
            self.assertNotIn("{{QUIZ_SCRIPT}}", content)
            self.assertNotIn("{{QUIZ_STYLE}}", content)
            self.assertNotIn("{{QUIZ_DATA}}", content)
            self.assertNotIn("https://github.com/Kairosxy/rl-infra-100days/issues", content)

    def test_every_related_course_number_is_valid(self):
        questions = read_bank(ROOT)["questions"]
        self.assertTrue(all(1 <= q["course_day"] <= 100 for q in questions))


class PublicRepositoryPrivacyTest(unittest.TestCase):
    def test_unpublished_personal_files_are_not_committed(self):
        for path in (
            "progress.md", "portfolio", "environments", "courses/day-01/answers.md",
            ".github/workflows/course-answer.yml", "scripts/issue_to_answer.py", ".env"
        ):
            self.assertFalse((ROOT / path).exists(), path)

    def test_quiz_saves_only_to_browser(self):
        js = (ROOT / "quiz" / "app.js").read_text(encoding="utf-8")
        self.assertIn("localStorage", js)
        self.assertNotIn("/issues/new", js)
        self.assertNotIn("fetch(", js)

    def test_course_page_has_no_issue_upload_logic(self):
        js = (ROOT / "site" / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("/issues/new", js)


if __name__ == "__main__":
    unittest.main()
