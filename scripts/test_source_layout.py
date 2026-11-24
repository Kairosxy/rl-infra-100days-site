"""Regression coverage for stable IDs, canonical article paths and cross-links."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from markdown_render import render_markdown
from source_layout import GROUPS, source_path, source_routes


ROOT = Path(__file__).resolve().parents[1]


class SourceLayoutTest(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((ROOT / "knowledge/catalog.json").read_text(encoding="utf-8"))
        self.chapters = {chapter["id"] for chapter in self.catalog["chapters"]}

    def test_every_article_has_one_canonical_source_and_stable_route(self):
        routes = source_routes(ROOT, self.catalog)
        self.assertEqual(len(routes), sum(len(self.catalog[group]) for group in GROUPS))
        for group in GROUPS:
            for item in self.catalog[group]:
                source = source_path(ROOT, group, item, self.chapters)
                self.assertEqual(source.stem, item["id"])
                self.assertEqual(len(source.relative_to(ROOT / "knowledge").parts), 3)
        self.assertIn("#topic=technology-lineage", routes.values())
        self.assertIn("#case=fsdp2-root-forward-oom", routes.values())
        self.assertIn("#implementation=fsdp2-sharding", routes.values())

    def test_reject_wrong_group_chapter_id_and_unsafe_paths(self):
        topic = self.catalog["topics"][0]
        for invalid in (None, "../quiz/data/questions.json", "/tmp/private.md",
                        "topics/../hidden.md", "cases/model/transformer-attention.md",
                        "topics/model/unrelated.md", "topics/rl/transformer-attention.md",
                        "topics//transformer-attention.md"):
            with self.subTest(source=invalid):
                item = {**topic, "source": invalid}
                with self.assertRaises(ValueError):
                    source_path(ROOT, "topics", item, self.chapters)

    def test_duplicate_source_fails(self):
        data = copy.deepcopy(self.catalog)
        data["topics"].append(data["topics"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate article source"):
            source_routes(ROOT, data)

    def test_unindexed_article_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            public = root / "knowledge/topics/model"
            public.mkdir(parents=True)
            (public / "unindexed.md").write_text("# Unindexed", encoding="utf-8")
            catalog = {"chapters": [{"id": "model"}], "topics": [],
                       "cases": [], "implementations": []}
            with self.assertRaisesRegex(ValueError, "Unindexed article"):
                source_routes(root, catalog)

    def test_article_link_uses_id_not_physical_path_and_keeps_fragment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "knowledge/topics/model/source.md"
            target = root / "knowledge/topics/rl/target.md"
            source.parent.mkdir(parents=True)
            target.parent.mkdir(parents=True)
            source.write_text("[Target](../rl/target.md#loss)", encoding="utf-8")
            target.write_text("# Target", encoding="utf-8")
            rendered = render_markdown(
                source, root, knowledge_routes={target: "#topic=target"}
            )
            self.assertIn('href="#topic=target&amp;anchor=loss"', rendered)

    def test_migrated_source_links_and_images(self):
        routes = source_routes(ROOT, self.catalog)
        source = ROOT / "knowledge/topics/optimization/fp8-numerics.md"
        rendered = render_markdown(source, ROOT, knowledge_routes=routes)
        self.assertIn('href="#topic=logprob-contract"', rendered)
        self.assertIn('href="#case=fp8-logprob-drift"', rendered)
        self.assertIn('href="#implementation=te-megatron-fp8"', rendered)
        self.assertIn("data:image/svg+xml;base64,", rendered)

    def test_broken_local_link_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.md"
            source.write_text("[Missing](missing.md)", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                render_markdown(source, Path(folder))
