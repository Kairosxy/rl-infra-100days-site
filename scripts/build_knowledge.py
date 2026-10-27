#!/usr/bin/env python3
"""Build the public knowledge reader.

Public topics, cases and implementation notes share one validated relation
graph. Archived courses and local answer records are built independently.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from markdown_render import render_markdown
from site_nav import NAV_CSS, nav_markup

GROUPS = ("topics", "cases", "implementations")
RELATIONS = {
    "topics": {"cases": "cases", "implementations": "implementations"},
    "cases": {"topics": "topics", "implementations": "implementations"},
    "implementations": {"topics": "topics", "cases": "cases"},
}


def knowledge_payload(root: Path) -> dict:
    payload = json.loads((root / "knowledge" / "catalog.json").read_text(encoding="utf-8"))
    if payload.get("schema") != "rl-infra-knowledge-v1":
        raise ValueError("Unknown knowledge catalog schema")

    ids_by_group: dict[str, set[str]] = {}
    for group in GROUPS:
        items = payload.get(group)
        if not isinstance(items, list):
            raise ValueError(f"Missing list {group}")
        identifiers = [item.get("id") for item in items]
        if any(not isinstance(i, str) or not re.fullmatch(r"[a-z0-9-]+", i) for i in identifiers):
            raise ValueError(f"Invalid ID in {group}")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Duplicate ID in {group}")
        ids_by_group[group] = set(identifiers)
        for item in items:
            source = root / "knowledge" / group / f'{item["id"]}.md'
            if not source.is_file():
                raise FileNotFoundError(source)
            if not item.get("title") or not item.get("summary"):
                raise ValueError(f"Missing description: {source}")
            item["html"] = render_markdown(source, root)

    chapter_ids = [item["id"] for item in payload.get("chapters", [])]
    if not chapter_ids or len(set(chapter_ids)) != len(chapter_ids):
        raise ValueError("Missing or duplicated chapters")
    for item in payload["topics"]:
        if item.get("chapter") not in chapter_ids:
            raise ValueError(f"Unknown chapter for topic {item['id']}")

    for group, relations in RELATIONS.items():
        for item in payload[group]:
            for relation, target_group in relations.items():
                for referenced_id in item.get(relation, []):
                    if referenced_id not in ids_by_group[target_group]:
                        raise ValueError(
                            f"Unknown relation {group}/{item['id']} "
                            f"-> {relation}/{referenced_id}"
                        )
    return payload


def generate_knowledge(root: Path, output: Path) -> int:
    root = root.resolve()
    catalog = knowledge_payload(root)
    bank = json.loads((root / "quiz" / "data" / "questions.json").read_text(encoding="utf-8"))
    questions = bank.get("questions", [])
    if bank.get("schema") != "rl-infra-question-bank-v1" or len(questions) != 224:
        raise ValueError("Knowledge product requires the complete 224-question bank")
    known_topics = {topic["id"] for topic in catalog["topics"]}
    for index, question in enumerate(questions, 1):
        if question.get("code") != f"Q{index:03d}" or question.get("topic") not in known_topics:
            raise ValueError(f"Invalid question linkage at Q{index:03d}")
    catalog["question_bank"] = questions
    site = root / "knowledge" / "site"
    html = (site / "index.html").read_text(encoding="utf-8")
    css = (site / "style.css").read_text(encoding="utf-8")
    js = (site / "app.js").read_text(encoding="utf-8")
    data = json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
    data = (data.replace("<", r"\u003c").replace(">", r"\u003e")
                .replace("&", r"\u0026").replace("\u2028", r"\u2028")
                .replace("\u2029", r"\u2029"))
    extra = ""
    for slot, value in {
        "{{SITE_NAV}}": nav_markup("knowledge", extra),
        "{{KNOWLEDGE_STYLE}}": css + "\n" + NAV_CSS,
        "{{KNOWLEDGE_DATA}}": data,
        "{{KNOWLEDGE_JS}}": js,
    }.items():
        if slot not in html:
            raise ValueError(f"Missing template slot {slot}")
        html = html.replace(slot, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    # Interactive labs are a supporting page of the knowledge site.
    labs_source = root / "labs"
    if labs_source.is_dir():
        labs_target = output.parent / "labs"
        labs_target.mkdir(parents=True, exist_ok=True)
        for asset in ("index.html", "style.css", "app.js", "parallel.html", "parallel-performance.html", "rl-memory.html", "workbench.css"):
            source_file = labs_source / asset
            if not source_file.is_file():
                raise FileNotFoundError(source_file)
            shutil.copy2(source_file, labs_target / asset)
        page = labs_target / "index.html"
        page.write_text(
            page.read_text(encoding="utf-8").replace("{{SITE_NAV}}", nav_markup("labs")),
            encoding="utf-8",
        )
        sheet = labs_target / "style.css"
        sheet.write_text(sheet.read_text(encoding="utf-8") + "\n" + NAV_CSS, encoding="utf-8")
    return sum(len(catalog[group]) for group in GROUPS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/index.html"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    count = generate_knowledge(root, args.output.resolve())
    print(f"Built {count} interconnected knowledge entries; no private data included.")


if __name__ == "__main__":
    main()
