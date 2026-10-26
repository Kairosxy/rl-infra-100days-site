"""Check rebuilt syllabus depth, interaction controls, and public content isolation."""
from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from build_site import course_payload, generate_site

ROOT = Path(__file__).resolve().parent.parent


class TeachingLabsTest(unittest.TestCase):
    def test_all_100_days_have_research_or_experiment_questions(self):
        text = (ROOT / "course_map" / "day_schedule.md").read_text(encoding="utf-8")
        matches = re.findall(r"^\|\s*(\d{2,3})\s*\|\s*[A-H]\s*\|\s*(?:深挖|实作|研读|综合)\s*\|", text, re.M)
        self.assertEqual([int(x) for x in matches], list(range(1, 101)))

    def test_first_five_are_aligned_to_new_sequence(self):
        titles = [x["title"] for x in course_payload(ROOT) if x["day"] <= 5]
        self.assertEqual(len(titles), 5)
        self.assertIn("SFT", titles[0])
        self.assertIn("梯度", titles[1])
        self.assertIn("Logprob", titles[2])
        self.assertIn("PPO", titles[3])
        self.assertIn("GRPO", titles[4])

    def test_labs_have_all_required_inputs_and_no_telemetry(self):
        html = (ROOT / "labs" / "index.html").read_text(encoding="utf-8")
        js = (ROOT / "labs" / "app.js").read_text(encoding="utf-8")
        for control in (
            "bandit-p", "bandit-n", "baseline-p", "baseline-b", "shift-p",
            "shift-wrong", "ppo-r", "ppo-a", "ppo-eps", "grpo-incomplete",
            "grpo-advantages",
        ):
            self.assertIn(f'id="{control}"', html)
        for day in range(1, 6):
            self.assertIn(f'id="lab-{day}"', html)
        self.assertNotIn("fetch(", js)
        self.assertNotIn("XMLHttpRequest", js)
        self.assertNotIn("localStorage", js)

    def test_site_build_copies_local_demos_and_hides_legacy_labs(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "index.html"
            generate_site(ROOT, dest)
            self.assertTrue((dest.parent / "labs" / "index.html").is_file())
            self.assertTrue((dest.parent / "labs" / "style.css").is_file())
            self.assertTrue((dest.parent / "labs" / "app.js").is_file())
            html = dest.read_text(encoding="utf-8")
            self.assertIn('href="labs/"', html)
            self.assertIn("labs/?day=", html)
        for course in course_payload(ROOT):
            if course["day"] <= 5:
                self.assertEqual(course["labs"], [],
                                 "Old example files must not be mislabeled by day")

    def test_old_static_figures_not_forced_into_new_lessons(self):
        for day in range(1, 6):
            root = ROOT / "courses" / f"day-{day:02d}"
            text = (root / "diagram.md").read_text(encoding="utf-8")
            self.assertIn(f"labs/?day={day}", text)
            self.assertNotIn("![](", text)


if __name__ == "__main__":
    unittest.main()
