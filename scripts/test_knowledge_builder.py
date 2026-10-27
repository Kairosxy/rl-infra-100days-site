"""Validate the public reader's data and routes, not incidental UI wording."""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from build_knowledge import generate_knowledge, knowledge_payload

ROOT = Path(__file__).resolve().parent.parent


class ReaderDocument(HTMLParser):
    """Inspect the generated document without executing its JavaScript."""

    def __init__(self):
        super().__init__()
        self.element_ids = []
        self.navigation = {}
        self.header_count = 0
        self.payload_chunks = []
        self.in_payload = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.element_ids.append(element_id)
        if tag == "header":
            self.header_count += 1
        if tag == "a" and attributes.get("data-nav"):
            self.navigation[attributes["data-nav"]] = attributes.get("href")
        if tag == "script" and element_id == "knowledge-data":
            self.in_payload = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_payload = False

    def handle_data(self, text):
        if self.in_payload:
            self.payload_chunks.append(text)


class KnowledgeReaderTest(unittest.TestCase):
    def test_graph_has_real_articles_and_valid_relationships(self):
        data = knowledge_payload(ROOT)
        self.assertEqual(data["schema"], "rl-infra-knowledge-v1")
        self.assertGreaterEqual(len(data["chapters"]), 6)
        self.assertGreaterEqual(len(data["topics"]), 10)
        self.assertGreaterEqual(len(data["cases"]), 5)
        self.assertGreaterEqual(len(data["implementations"]), 6)
        for collection in ("topics", "cases", "implementations"):
            for item in data[collection]:
                self.assertIn("html", item)
                self.assertIn("<p>", item["html"])
                self.assertGreater(len(item["html"]), 300)

    def test_root_reader_serves_the_knowledge_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "index.html"
            count = generate_knowledge(ROOT, output)
            html = output.read_text(encoding="utf-8")
            self.assertGreaterEqual(count, 25)
            for slot in ("KNOWLEDGE_STYLE", "KNOWLEDGE_DATA", "KNOWLEDGE_JS"):
                self.assertFalse("{{" + slot + "}}" in html, f"Unresolved template slot: {slot}")

            document = ReaderDocument()
            document.feed(html)
            document.close()
            for element_id in ("knowledge-data", "home", "chapter", "questions", "detail", "search", "article"):
                self.assertIn(element_id, document.element_ids)
            duplicates = [key for key, value in Counter(document.element_ids).items() if value > 1]
            self.assertEqual(duplicates, [], "Duplicate IDs can break reader navigation")
            self.assertEqual(document.header_count, 1, "The root reader must have one header")
            self.assertEqual(document.navigation.get("map"), "#")
            self.assertEqual(document.navigation.get("questions"), "#view=questions")
            self.assertEqual(document.navigation.get("library"), "#view=library")

            self.assertTrue(document.payload_chunks, "Missing embedded knowledge data")
            payload = json.loads("".join(document.payload_chunks))
            self.assertEqual(payload["schema"], "rl-infra-knowledge-v1")
            self.assertEqual(count, sum(len(payload[group]) for group in ("topics", "cases", "implementations")))
            topic_ids = {topic["id"] for topic in payload["topics"]}
            self.assertIn("fp8-numerics", topic_ids)
            questions = payload.get("question_bank", [])
            self.assertEqual(len(questions), 224)
            self.assertEqual([question["code"] for question in questions], [f"Q{index:03d}" for index in range(1, 225)])
            for question in questions:
                self.assertIn(question["topic"], topic_ids, question["code"])
                for field in ("question", "brief", "deep", "followup"):
                    self.assertTrue(question.get(field), f"Missing {field} in {question['code']}")

            for forbidden in ("legacy.html", 'id="answer-q1"', "RL_INFRA_ANSWER_V1", "\u6211\u7684\u4f5c\u7b54"):
                self.assertFalse(forbidden in html, f"Unexpected legacy or personal content: {forbidden}")

    def test_build_creates_missing_output_directories_and_copies_labs(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "fresh" / "dist" / "index.html"
            self.assertFalse(output.parent.exists())
            generate_knowledge(ROOT, output)
            self.assertTrue(output.is_file())
            for asset in ("index.html", "style.css", "app.js", "parallel.html", "parallel-performance.html", "rl-memory.html", "workbench.css"):
                path = output.parent / "labs" / asset
                self.assertTrue(path.is_file(), f"Missing published lab asset: {asset}")
                self.assertGreater(path.stat().st_size, 0)
            lab_html = (output.parent / "labs" / "index.html").read_text(encoding="utf-8")
            self.assertFalse("{{SITE_NAV}}" in lab_html, "Lab navigation was not rendered")

    def test_no_filler_questions_and_all_three_content_types(self):
        data = knowledge_payload(ROOT)
        kinds = {key: len(data[key]) for key in ("topics", "cases", "implementations")}
        self.assertTrue(all(kinds.values()))
        self.assertTrue(all(item.get("recall", {}).get("question") for item in data["topics"]))
        self.assertNotIn("\u8bf7\u89e3\u91ca\u672c\u8bfe\u7684\u6838\u5fc3\u673a\u5236", (ROOT / "knowledge" / "site" / "index.html").read_text(encoding="utf-8"))

    def test_source_paths_cannot_be_arbitrary_user_records(self):
        data = json.loads((ROOT / "knowledge" / "catalog.json").read_text(encoding="utf-8"))
        valid = re.compile(r"^[a-z0-9-]+$")
        for group in ("topics", "cases", "implementations"):
            for item in data[group]:
                self.assertIsNotNone(valid.fullmatch(item["id"]))
                self.assertNotIn("answers", item["id"])


if __name__ == "__main__":
    unittest.main()
