"""The knowledge graph is generated from public content, never personal notes."""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from build_knowledge import generate_knowledge, knowledge_payload

ROOT=Path(__file__).resolve().parent.parent


class KnowledgeReaderTest(unittest.TestCase):
    def test_graph_has_real_articles_and_valid_relationships(self):
        data=knowledge_payload(ROOT)
        self.assertEqual(data["schema"],"rl-infra-knowledge-v1")
        self.assertGreaterEqual(len(data["chapters"]),6)
        self.assertGreaterEqual(len(data["topics"]),10)
        self.assertGreaterEqual(len(data["cases"]),5)
        self.assertGreaterEqual(len(data["implementations"]),6)
        for collection in ("topics","cases","implementations"):
            for item in data[collection]:
                self.assertIn("html",item)
                self.assertIn("<p>",item["html"])
                self.assertGreater(len(item["html"]),300)

    def test_root_reader_and_archived_day_links_are_both_available(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"index.html"
            count=generate_knowledge(ROOT,output)
            html=output.read_text(encoding="utf-8")
            self.assertGreaterEqual(count,25)
            self.assertNotIn("{{KNOWLEDGE_DATA}}",html)
            self.assertNotIn("{{KNOWLEDGE_JS}}",html)
            self.assertIn("基础知识",html)
            self.assertIn("FP8",html)
            self.assertIn("legacy.html",html)
            self.assertIn('location.replace("legacy.html"+location.hash)',html)
            self.assertNotIn('id="answer-q1"',html)
            self.assertNotIn('RL_INFRA_ANSWER_V1',html)
            self.assertNotIn("我的作答",html)

    def test_no_filler_questions_and_all_three_content_types(self):
        data=knowledge_payload(ROOT)
        kinds={k:len(data[k]) for k in ("topics","cases","implementations")}
        self.assertTrue(all(kinds.values()))
        self.assertTrue(all(item.get("recall",{}).get("question")
                            for item in data["topics"]))
        self.assertNotIn("请解释本课的核心机制",
                         (ROOT/"knowledge"/"site"/"index.html").read_text(encoding="utf-8"))

    def test_source_paths_cannot_be_arbitrary_user_records(self):
        data=json.loads((ROOT/"knowledge"/"catalog.json").read_text(encoding="utf-8"))
        valid=re.compile(r"^[a-z0-9-]+$")
        for group in ("topics","cases","implementations"):
            for item in data[group]:
                self.assertIsNotNone(valid.fullmatch(item["id"]))
                self.assertNotIn("answers",item["id"])


if __name__=="__main__":
    unittest.main()
