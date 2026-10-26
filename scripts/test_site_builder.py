"""CPU-only checks for the generated course reader."""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from build_site import course_payload, generate_site, render_lesson_markdown


ROOT = Path(__file__).resolve().parent.parent


class CourseWebsiteTest(unittest.TestCase):
    def test_available_lessons_are_indexed(self):
        pages = course_payload(ROOT)
        days = {int(page["day"]) for page in pages}
        self.assertTrue({1, 2, 3, 4, 5}.issubset(days))
        self.assertEqual(len(days), len(pages))

    def test_private_answers_are_not_embedded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "index.html"
            count = generate_site(ROOT, output)
            content = output.read_text(encoding="utf-8")
            self.assertGreaterEqual(count, 5)
            self.assertNotIn("{{COURSE_DATA}}", content)
            self.assertNotIn("{{STYLE}}", content)
            self.assertNotIn("{{APP_JS}}", content)
            self.assertNotIn("生成器只提供问题和结构，不替学习者编写经历", content)
            self.assertNotIn("本文件由学习者填写", content)
            self.assertIn("交互实验室", content)
            self.assertTrue((output.parent / "labs" / "index.html").is_file())
            self.assertTrue((output.parent / "labs" / "app.js").is_file())
            self.assertIn("PPO Clipped Surrogate", (output.parent / "labs" / "index.html").read_text(encoding="utf-8"))
            self.assertIn("Day 04", content)
            self.assertIn("RL_INFRA_ANSWER_V1", content)

    def test_math_and_code_survive_html_conversion(self):
        source = ROOT / "courses" / "day-03" / "lesson.md"
        html = render_lesson_markdown(source, ROOT)
        self.assertIn("arithmatex", html)
        self.assertIn("language-", html)
        self.assertIn("math", html.lower())

    def test_questions_present_for_every_published_lesson(self):
        for item in course_payload(ROOT):
            self.assertEqual(len(item["questions"]), 2)
            self.assertTrue(item["title"].strip())


if __name__ == "__main__":
    unittest.main()
