"""Protect public Pages source from accidentally incorporating private learning files."""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_PATHS = (
    "progress.md",
    "portfolio",
    "environments",
    ".env",
    ".env.local",
    "scripts/issue_to_answer.py",
    ".github/workflows/course-answer.yml",
)
PRIVATE_ANSWER_FILENAMES = {"answers.md", "answers.json", "learner_progress.json"}


class PublicBoundaryTest(unittest.TestCase):
    def test_no_private_learning_files_exist_in_public_repo(self):
        for relative in FORBIDDEN_PATHS:
            self.assertFalse((ROOT / relative).exists(), relative)
        for path in ROOT.rglob("*"):
            if path.is_file() and (".git" not in path.parts):
                self.assertNotIn(path.name.lower(), PRIVATE_ANSWER_FILENAMES, path.as_posix())

    def test_knowledge_reader_has_no_write_endpoints(self):
        text = (ROOT / "knowledge" / "site" / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("/issues/new", text)
        self.assertNotIn("fetch(", text)
        self.assertNotIn("XMLHttpRequest", text)

    def test_legacy_course_reader_is_gone(self):
        for relative in ("courses", "site", "scripts/build_site.py"):
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_quiz_only_stores_progress_in_browser(self):
        text = (ROOT / "quiz" / "app.js").read_text(encoding="utf-8")
        self.assertIn("localStorage", text)
        self.assertNotIn("fetch(", text)
        self.assertNotIn("XMLHttpRequest", text)


if __name__ == "__main__":
    unittest.main()
