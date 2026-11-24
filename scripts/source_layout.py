"""Resolve canonical knowledge sources independently of their public route IDs."""
from __future__ import annotations

import re
from pathlib import Path

GROUPS = ("topics", "cases", "implementations")
ROUTE_KIND = {"topics": "topic", "cases": "case", "implementations": "implementation"}
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def source_path(root: Path, group: str, item: dict, chapter_ids: set[str]) -> Path:
    """Accept only group/domain/id.md paths under the public knowledge tree."""
    if group not in GROUPS:
        raise ValueError(f"Unknown content group: {group}")
    identifier = item.get("id")
    source = item.get("source")
    if not isinstance(identifier, str) or not SLUG.fullmatch(identifier):
        raise ValueError(f"Invalid content ID: {identifier!r}")
    if not isinstance(source, str):
        raise ValueError(f"Missing source path: {group}/{identifier}")
    parts = source.split("/")
    if (len(parts) != 3 or parts[0] != group
            or not SLUG.fullmatch(parts[1]) or parts[2] != f"{identifier}.md"):
        raise ValueError(f"Invalid source path: {source!r}")
    domain = parts[1]
    if group == "topics" and domain != item.get("chapter"):
        raise ValueError(f"Topic source does not match chapter: {source}")
    if group in ("topics", "cases") and domain not in chapter_ids:
        raise ValueError(f"Unknown source domain: {source}")
    knowledge = root.resolve() / "knowledge"
    candidate = knowledge / source
    if candidate.resolve() != candidate or not candidate.is_file():
        raise ValueError(f"Missing or noncanonical public source: {source}")
    return candidate


def source_routes(root: Path, catalog: dict) -> dict[Path, str]:
    """Create the only source-to-route index; reject duplicate or unindexed articles."""
    chapters = {chapter["id"] for chapter in catalog["chapters"]}
    routes: dict[Path, str] = {}
    for group in GROUPS:
        for item in catalog[group]:
            source = source_path(root, group, item, chapters)
            if source in routes:
                raise ValueError(f"Duplicate article source: {source}")
            routes[source] = f"#{ROUTE_KIND[group]}={item['id']}"

    for group in GROUPS:
        for source in (root.resolve() / "knowledge" / group).rglob("*.md"):
            if source not in routes:
                raise ValueError(f"Unindexed article: {source}")
    return routes
