#!/usr/bin/env python3
"""Build standalone interview question-bank page from reviewed public JSON.

Only bundled educational questions and reference explanations are published.
Do not load or bundle learner answers, progress, private project notes, or tokens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TOTAL_QUESTIONS = 224


def get_question_bank(root: Path) -> dict:
    source = root / "quiz" / "data" / "questions.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    items = payload.get("questions", [])
    if payload.get("schema") != "rl-infra-question-bank-v1":
        raise ValueError("Unsupported question bank schema")
    if len(items) != TOTAL_QUESTIONS or payload.get("question_count") != TOTAL_QUESTIONS:
        raise ValueError("The question bank must have 224 numbered questions")
    for index, question in enumerate(items, start=1):
        if (
            question.get("id") != index
            or question.get("code") != f"Q{index:03d}"
            or not all(question.get(k) for k in ("question", "brief", "deep", "followup"))
        ):
            raise ValueError(f"Invalid or incomplete question Q{index:03d}")
    return payload


def generate_quiz(root: Path, output: Path) -> int:
    root = root.resolve()
    template = (root / "quiz" / "template.html").read_text(encoding="utf-8")
    css = (root / "quiz" / "style.css").read_text(encoding="utf-8")
    js = (root / "quiz" / "app.js").read_text(encoding="utf-8")
    # Avoid UTC shifting the browser's local due date to yesterday in UTC+ time zones.
    js = js.replace(
        'return now.toISOString().slice(0, 10);',
        'return [now.getFullYear(), two(now.getMonth() + 1), two(now.getDate())].join("-");',
    )
    payload = get_question_bank(root)
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data_json = (data_json.replace("<", r"\u003c").replace(">", r"\u003e")
                 .replace("&", r"\u0026").replace("\u2028", r"\u2028")
                 .replace("\u2029", r"\u2029"))
    replacements = {
        "{{QUIZ_STYLE}}": css,
        "{{QUIZ_DATA}}": data_json,
        "{{QUIZ_SCRIPT}}": js,
    }
    for placeholder, value in replacements.items():
        if placeholder not in template:
            raise ValueError(f"Missing template slot {placeholder}")
        template = template.replace(placeholder, value)
    if "<script type=\"application/json\"" not in template:
        raise ValueError("Missing JSON data wrapper")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template, encoding="utf-8")
    return TOTAL_QUESTIONS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/quiz/index.html"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    count = generate_quiz(root, args.output.resolve())
    print(f"Built question bank: {count} questions; browser-local progress only.")


if __name__ == "__main__":
    main()
