#!/usr/bin/env python3
"""Shared top navigation for the knowledge site, labs and quiz.

Both builders inject the same markup and CSS so the three pages look like one
site. Keeping it here avoids three hand-edited copies drifting apart.
"""
from __future__ import annotations

# One row on every page: wordmark on the left, the same three destinations on
# the right. `aria-current` marks the page the reader is on.
NAV_HTML = """<header class="site-nav">
    <a class="site-nav-brand" href="{home}">RL Infra<span class="site-nav-dot">.</span></a>
    <nav aria-label="网站导航" class="site-nav-links">
      <a href="{home}"{current_knowledge}>知识库</a>
      <a href="{labs}"{current_labs}>实验</a>
      <a href="{quiz}"{current_quiz}>复习</a>
    </nav>
    {extra}
  </header>"""

NAV_CSS = """.site-nav{height:64px;display:flex;align-items:center;justify-content:space-between;
  gap:24px;padding:0 max(18px,calc((100vw - 1420px)/2));border-bottom:1px solid #e4e7e1;
  background:rgba(251,251,248,.97);position:sticky;top:0;z-index:30}
.site-nav-brand{font:700 20px Georgia,serif;letter-spacing:-.04em;color:#2e4036;
  text-decoration:none;white-space:nowrap}
.site-nav-brand:hover{text-decoration:none;color:#2e4036}
.site-nav-dot{color:#a65e45}
.site-nav-links{display:flex;align-items:center;gap:26px;margin-left:auto}
.site-nav-links a{font-size:13px;color:#596a60;text-decoration:none;white-space:nowrap}
.site-nav-links a:hover{color:#26523a;text-decoration:none}
.site-nav-links a[aria-current="page"]{color:#26523a;font-weight:700}
@media(max-width:640px){.site-nav{height:58px;padding:0 15px;gap:12px}
  .site-nav-brand{font-size:18px}.site-nav-links{gap:16px}.site-nav-links a{font-size:12.5px}}"""


def nav_markup(current: str, extra: str = "") -> str:
    """Return the shared header. `current` is knowledge, labs or quiz."""
    def mark(name: str) -> str:
        return ' aria-current="page"' if name == current else ""
    return NAV_HTML.format(
        home="../" if current != "knowledge" else "./",
        labs="../labs/" if current != "knowledge" else "labs/",
        quiz="../quiz/" if current != "knowledge" else "quiz/",
        current_knowledge=mark("knowledge"),
        current_labs=mark("labs"),
        current_quiz=mark("quiz"),
        extra=extra,
    )
