#!/usr/bin/env python3
"""Build the public, static interview question bank from reviewed study material.

All learner answers and review progress remain in browser localStorage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from site_nav import NAV_CSS, nav_markup

TOTAL_QUESTIONS = 224


def read_bank(root: Path) -> dict:
    content = json.loads((root / "quiz" / "data" / "questions.json").read_text(encoding="utf-8"))
    questions = content.get("questions", [])
    if content.get("schema") != "rl-infra-question-bank-v1" or len(questions) != TOTAL_QUESTIONS:
        raise ValueError("Expected a complete 224-question bank")
    for index, q in enumerate(questions, start=1):
        if q.get("id") != index or q.get("code") != f"Q{index:03d}":
            raise ValueError(f"Unexpected question numbering at item {index}")
        if not all(q.get(k) for k in ("question", "brief", "deep", "followup", "domain")):
            raise ValueError(f"Missing text for Q{index:03d}")
        if q.get("level") not in ("基础", "进阶"):
            raise ValueError(f"Unrecognized difficulty for Q{index:03d}")
    return content


def build(root: Path, output: Path) -> int:
    root = root.resolve()
    template = (root / "quiz" / "template.html").read_text(encoding="utf-8")
    css = (root / "quiz" / "style.css").read_text(encoding="utf-8")
    js = (root / "quiz" / "app.js").read_text(encoding="utf-8")
    bank = read_bank(root)
    catalog = json.loads((root / "knowledge" / "catalog.json").read_text(encoding="utf-8"))
    known = {item["id"] for item in catalog["topics"]}
    for question in bank["questions"]:
        if question.get("topic") not in known:
            raise ValueError(f"Unknown topic for {question['code']}: {question.get('topic')}")
    bank["topics"] = [{"id": item["id"], "title": item["title"]} for item in catalog["topics"]]
    data = json.dumps(bank, ensure_ascii=False, separators=(",", ":"))
    # Avoid an accidental closing script tag in the embedded JSON.
    data = (data.replace("<", r"\u003c").replace(">", r"\u003e")
                .replace("&", r"\u0026").replace("\u2028", r"\u2028")
                .replace("\u2029", r"\u2029"))
    replacements = {
        "{{SITE_NAV}}": nav_markup("quiz"),
        "{{QUIZ_STYLE}}": css + "\n" + NAV_CSS, "{{QUIZ_DATA}}": data, "{{QUIZ_SCRIPT}}": js
    }
    for key, value in replacements.items():
        if key not in template:
            raise ValueError(f"Missing template slot {key}")
        template = template.replace(key, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template, encoding="utf-8")
    return TOTAL_QUESTIONS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/quiz/index.html"))
    args = parser.parse_args()
    count = build(Path(__file__).resolve().parent.parent, args.output.resolve())
    print(f"Built {args.output}: {count} questions, no learner data.")


if __name__ == "__main__":
    main()
