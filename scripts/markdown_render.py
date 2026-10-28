#!/usr/bin/env python3
"""Render repository Markdown into self-contained HTML.

Shared by the knowledge builder. Local images are inlined as data URIs and
local links are rewritten to the public repository so the page works when it
is served from GitHub Pages.
"""
from __future__ import annotations

import base64
import html
import mimetypes
import re
from pathlib import Path
from urllib.parse import unquote, urlparse, quote

import markdown
from pymdownx import slugs

REPOSITORY = "Kairosxy/rl-infra-100days-site"
IMAGE_RE = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")', re.IGNORECASE)
LINK_RE = re.compile(r'(<a\b[^>]*?\bhref=")([^"]+)(")', re.IGNORECASE)
LEADING_H1_RE = re.compile(r"^\s*<h1\b[^>]*>.*?</h1>\s*", re.DOTALL)


def local_reference(path: str, source: Path, root: Path) -> Path | None:
    """Resolve references without allowing traversal outside the repo root."""
    parsed = urlparse(html.unescape(path))
    if parsed.scheme or path.startswith("#") or path.startswith("//"):
        return None
    resolved = (source.parent / unquote(parsed.path)).resolve()
    if not resolved.is_relative_to(root.resolve()):
        return None
    return resolved


def render_markdown(source: Path, root: Path, drop_title: bool = True) -> str:
    """Render GFM tables, code blocks and MathJax-ready formulas."""
    rendered = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "pymdownx.arithmatex", "toc"],
        extension_configs={
            "pymdownx.arithmatex": {"generic": True},
            # Keep Chinese heading text in ids so anchors survive inserted sections.
            "toc": {"slugify": slugs.slugify(case="lower", percent_encode=False)},
        },
        output_format="html5",
    )
    if drop_title:
        # The page header already shows the entry title.
        rendered = LEADING_H1_RE.sub("", rendered, count=1)

    def inline_image(match: re.Match[str]) -> str:
        referenced = local_reference(match.group(2), source, root)
        if referenced is None or not referenced.is_file():
            return match.group(0)
        mime = mimetypes.guess_type(referenced.name)[0]
        if mime not in ("image/svg+xml", "image/png", "image/jpeg", "image/webp"):
            return match.group(0)
        encoded = base64.b64encode(referenced.read_bytes()).decode("ascii")
        return f'{match.group(1)}data:{mime};base64,{encoded}{match.group(3)}'

    rendered = IMAGE_RE.sub(inline_image, rendered)

    def link_to_source(match: re.Match[str]) -> str:
        referenced = local_reference(match.group(2), source, root)
        if referenced is None or not referenced.is_file():
            return match.group(0)
        relative = referenced.relative_to(root.resolve()).as_posix()
        fragment = urlparse(html.unescape(match.group(2))).fragment
        destination = f"https://github.com/{REPOSITORY}/blob/main/{relative}"
        parts = Path(relative).parts
        if len(parts) == 3 and parts[0] == "knowledge" and parts[1] in ("topics", "cases", "implementations"):
            kind = {"topics": "topic", "cases": "case", "implementations": "implementation"}[parts[1]]
            destination = f"#{kind}={Path(relative).stem}"
            if fragment:
                destination += "&anchor=" + quote(fragment)
        elif parts[0] == "labs":
            destination = relative + ("#" + fragment if fragment else "")
        elif fragment:
            destination += "#" + fragment
        return f'{match.group(1)}{html.escape(destination, quote=True)}{match.group(3)}'

    return LINK_RE.sub(link_to_source, rendered)
