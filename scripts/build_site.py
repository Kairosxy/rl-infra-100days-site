#!/usr/bin/env python3
"""Build a standalone HTML reader from repository lessons.

Do not publish learner answers or private project notes. GitHub Pages sites
can be publicly accessible even when the source repository is private.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

import markdown


REPOSITORY = "Kairosxy/rl-infra-100days-site"
SECTIONS = ("lesson", "diagram", "recap", "qa")
IMAGE_RE = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")', re.IGNORECASE)
LINK_RE = re.compile(r'(<a\b[^>]*?\bhref=")([^"]+)(")', re.IGNORECASE)


def local_reference(path: str, source: Path, root: Path) -> Path | None:
    """Resolve references without allowing traversal outside the repo root."""
    parsed = urlparse(html.unescape(path))
    if parsed.scheme or path.startswith("#") or path.startswith("//"):
        return None
    resolved = (source.parent / unquote(parsed.path)).resolve()
    if not resolved.is_relative_to(root.resolve()):
        return None
    return resolved


def render_lesson_markdown(source: Path, root: Path) -> str:
    """Render GFM-style tables, code blocks and MathJax-ready formulas."""
    rendered = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "pymdownx.arithmatex", "toc"],
        extension_configs={"pymdownx.arithmatex": {"generic": True}},
        output_format="html5",
    )

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
        if fragment:
            destination += "#" + fragment
        return f'{match.group(1)}{html.escape(destination, quote=True)}{match.group(3)}'

    return LINK_RE.sub(link_to_source, rendered)


def course_payload(root: Path) -> list[dict[str, object]]:
    """Include published lesson content but never embed personal answers."""
    courses = []
    for day in range(1, 101):
        folder = root / "courses" / f"day-{day:02d}"
        lesson_file = folder / "lesson.md"
        if not lesson_file.exists():
            continue

        original = lesson_file.read_text(encoding="utf-8")
        match = re.search(r"^#\s+Day\s*\d+(?:/100)?\s*[·:：\-]?\s*(.+)$", original, re.M)
        title = match.group(1).strip() if match else f"Day {day:02d}"
        sections = {
            section: render_lesson_markdown(folder / f"{section}.md", root)
            if (folder / f"{section}.md").exists()
            else ""
            for section in SECTIONS
        }

        qa = folder / "qa.md"
        questions = (
            re.findall(r"^##\s*(题\s*[12][^\n]*)", qa.read_text(encoding="utf-8"), re.M)
            if qa.exists()
            else []
        )
        while len(questions) < 2:
            questions.append(
                "请解释本课的核心机制。" if not questions else "给出一个失败案例和验证方法。"
            )

        # Legacy day_01..day_04 code is retained for later topics in the new syllabus.
        # Avoid labeling these old experiments as the current lesson's lab.
        lab_folder = root / "examples" / f"day_{day:02d}" if day > 5 else None
        labs = [
            path.relative_to(root).as_posix()
            for path in sorted(lab_folder.glob("*.py"))
            if path.is_file()
        ] if lab_folder is not None and lab_folder.exists() else []

        courses.append({
            "day": day,
            "title": title,
            "sections": sections,
            "questions": questions[:2],
            "labs": labs,
        })
    return courses


def generate_site(root: Path, output: Path) -> int:
    """Generate a one-file HTML reader with embedded curriculum and pictures."""
    root = root.resolve()
    source = (root / "site" / "index.html").read_text(encoding="utf-8")
    css = (root / "site" / "style.css").read_text(encoding="utf-8")
    js = (root / "site" / "app.js").read_text(encoding="utf-8")
    payload = json.dumps(course_payload(root), ensure_ascii=False, separators=(",", ":"))
    # Keep all user-independent JSON inside a script[type=application/json].
    payload = payload.replace("<", r"\u003c").replace(">", r"\u003e").replace("&", r"\u0026")
    tokens = {"{{STYLE}}": css, "{{COURSE_DATA}}": payload, "{{APP_JS}}": js}
    for slot, value in tokens.items():
        if slot not in source:
            raise ValueError(f"Missing slot {slot} in site/index.html")
        source = source.replace(slot, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")
    # Publish interactive demos alongside the standalone course reader.
    labs_source = root / "labs"
    if labs_source.is_dir():
        labs_target = output.parent / "labs"
        labs_target.mkdir(parents=True, exist_ok=True)
        for asset in ("index.html", "style.css", "app.js"):
            source_file = labs_source / asset
            if not source_file.is_file():
                raise FileNotFoundError(f"Missing interactive lab asset: {source_file}")
            shutil.copy2(source_file, labs_target / asset)
    return len(json.loads(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/index.html"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    count = generate_site(root, args.output.resolve())
    print(f"Built {args.output} with {count} lessons; private answers excluded.")


if __name__ == "__main__":
    main()
