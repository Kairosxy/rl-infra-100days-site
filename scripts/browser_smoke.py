#!/usr/bin/env python3
"""Exercise the generated product in Chromium, including real browser storage in CI.

--inline-preview is solely for environments where browser navigation is blocked.
It verifies DOM/JS with an in-memory storage substitute and is NOT deployment QA.
The default mode serves dist over HTTP and uses actual localStorage.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import re
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

KEY = 'rl-infra-100days-questionbank-v1'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', type=Path, default=Path('dist'))
    parser.add_argument('--output', type=Path, default=Path('test-artifacts'))
    parser.add_argument('--inline-preview', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    html = (args.dist / 'index.html').read_text(encoding='utf-8')
    payload = json.loads(re.search(r'<script type="application/json" id="knowledge-data">(.*?)</script>', html, re.S).group(1))
    entries = [(kind, x['id'], x['title']) for kind, name in [('topic', 'topics'), ('case', 'cases'), ('implementation', 'implementations')] for x in payload[name]]
    questions = [('question', q['code'], q['question']) for q in payload['question_bank']]
    report = {'mode': 'inline-preview-with-storage-substitute' if args.inline_preview else 'http-real-localStorage', 'routes': 0, 'experiments': 0, 'viewports': [], 'checks': []}
    server = None
    if not args.inline_preview:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(args.dist.resolve()))
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f'http://127.0.0.1:{server.server_port}/'
    with sync_playwright() as p:
        executable = os.environ.get('CHROMIUM_EXECUTABLE')
        browser = p.chromium.launch(**({'executable_path': executable} if executable else {}))
        context = browser.new_context(viewport={'width': 1440, 'height': 1000}, accept_downloads=True)
        page = context.new_page()
        page.set_default_timeout(10000)
        errors: list[str] = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        # Optional CDN renderers may be unavailable. Core navigation/experiments must work without them.
        page.route('https://cdn.jsdelivr.net/**', lambda route: route.abort())
        if args.inline_preview:
            html = re.sub(r'<script[^>]+src=[^>]*>\s*</script>', '', html)
            html = re.sub(r'<script type="module">.*?</script>', '', html, flags=re.S)
            shim = "<script>const store={};Object.defineProperty(window,'localStorage',{value:{getItem:k=>store[k]??null,setItem:(k,v)=>store[k]=String(v),removeItem:k=>delete store[k],clear:()=>Object.keys(store).forEach(k=>delete store[k])}});</script>"
            page.set_content(html.replace('<body>', '<body>' + shim), wait_until='domcontentloaded')
        else:
            page.goto(base, wait_until='domcontentloaded')

        def go(fragment: str) -> None:
            page.evaluate('(h)=>{if(location.hash===h)dispatchEvent(new HashChangeEvent("hashchange"));else location.hash=h;}', fragment)
            page.wait_for_timeout(25)

        def check(condition: bool, name: str) -> None:
            if not condition:
                raise AssertionError(name)
            report['checks'].append(name)

        try:
            check(page.locator('header').count() == 1, 'one global navigation header')
            check(len(questions) == 224, '224 original questions in rendered product')
            for kind, ident, title in entries + questions:
                go(f'#{kind}={ident}')
                check(page.locator('#detail-title').inner_text() == title, f'route {kind}:{ident}')
                check(page.locator('#detail').is_visible(), f'visible {kind}:{ident}')
                report['routes'] += 1
            check(not errors, 'all content routes without runtime exceptions')
            for spec in payload['learning_map']['experiments']:
                kind, ident = spec['entry'].split(':', 1)
                go(f'#{kind}={ident}')
                widget = page.locator(f'[data-experiment="{spec["type"]}"]')
                check(widget.count() == 1, 'native experiment ' + spec['type'])
                check(widget.locator('.experiment-output').inner_text().strip() != '', 'nonempty model ' + spec['type'])
                select = widget.locator('select').first
                if select.count():
                    options = select.locator('option').evaluate_all('(os)=>os.map(o=>o.value)')
                    select.select_option(options[-1])
                    check(widget.locator('.experiment-output').inner_text().strip() != '', 'updated model ' + spec['type'])
                report['experiments'] += 1
            check(page.locator('iframe').count() == 0, 'interactions belong to article, not embedded web pages')
            # Guard the exact prior failure: destroy an interactive article, then open another.
            for fragment in ['#topic=tp-matrix-communication', '#topic=pp-pipeline-bubble', '#question=Q081', '#topic=tp-matrix-communication']:
                go(fragment)
            check(not errors, 'interactive article lifecycle regression')
            go('#view=questions&kind=question')
            check('224' in page.locator('#questions .count').inner_text(), 'question filter includes all 224')
            page.locator('#search').fill('Q081')
            page.wait_for_timeout(220)
            check(page.locator('#questions .entry').count() == 1, 'search stays within question center')
            page.locator('#questions .entry').click()
            check(page.locator('#detail-title').inner_text() == questions[80][2], 'search result deep link')
            go('#view=questions&area=parallel&section=parallel-2')
            check(page.locator('#questions .entry').count() > 0, 'section-scoped questions are not empty')
            # Preserve legacy drafts and independent records while rating in the new product.
            original = {'version': 1, 'records': {'81': {'answer': 'saved-draft', 'layers': {'c1': 'saved-layer'}, 'note': 'saved-note', 'rating': 'hard', 'attempts': 1, 'streak': 0, 'due': '2026-01-01', 'bookmarked': True, 'history': [{'rating': 'hard', 'date': '2026-01-01T12:00:00Z'}]}, '2': {'answer': 'other-record'}}}
            page.evaluate('([key,value])=>localStorage.setItem(key,JSON.stringify(value))', [KEY, original])
            go('#question=Q081')
            page.locator('[data-rating="good"]').click()
            stored = page.evaluate('(key)=>JSON.parse(localStorage.getItem(key))', KEY)
            check(stored['records']['81']['answer'] == 'saved-draft' and stored['records']['81']['layers']['c1'] == 'saved-layer', 'rating preserves legacy draft and layers')
            check(stored['records']['2']['answer'] == 'other-record', 'rating preserves other questions')
            check(stored['records']['81']['attempts'] == 2 and stored['records']['81']['rating'] == 'good', 'rating writes same legacy storage schema')
            go('#view=questions&review=bookmarked')
            check(page.locator('#questions .entry').count() == 1, 'bookmark filter')
            incoming = {'format': 'rl-infra-interview-practice-v1', 'records': {'81': {'note': 'must-not-overwrite'}, '3': {'note': 'imported'}}}
            page.locator('#questions input[type=file]').set_input_files({'name': 'records.json', 'mimeType': 'application/json', 'buffer': json.dumps(incoming).encode()})
            page.wait_for_timeout(80)
            stored = page.evaluate('(key)=>JSON.parse(localStorage.getItem(key))', KEY)
            check(stored['records']['81']['note'] == 'saved-note' and stored['records']['3']['note'] == 'imported', 'import merges without overwriting existing records')
            incoming['records']['999'] = {'rating': 'easy'}
            page.locator('#questions input[type=file]').set_input_files({'name': 'invalid.json', 'mimeType': 'application/json', 'buffer': json.dumps(incoming).encode()})
            page.wait_for_timeout(80)
            check(page.evaluate('(key)=>JSON.parse(localStorage.getItem(key))', KEY) == stored, 'invalid import is atomic and does not change stored data')
            export_button = page.locator('#questions .review-tools button').first
            with page.expect_download() as download_info:
                export_button.click()
            exported = download_info.value
            exported.save_as(args.output / 'review-roundtrip.json')
            backup = json.loads((args.output / 'review-roundtrip.json').read_text())
            check(backup['records'] == stored['records'], 'export preserves records for old quiz import')
            if not args.inline_preview:
                page.reload(wait_until='domcontentloaded')
                check(page.evaluate('(key)=>JSON.parse(localStorage.getItem(key))', KEY) == stored, 'real localStorage persists across reload')
                page.goto(base + 'quiz/?q=81', wait_until='domcontentloaded')
                check(page.evaluate('(key)=>JSON.parse(localStorage.getItem(key))', KEY) == stored, 'legacy quiz reads unchanged storage')
                page.goto(base, wait_until='domcontentloaded')
            page.evaluate('(key)=>localStorage.setItem(key,"not-valid-json")', KEY)
            go('#question=Q081')
            check(page.locator('[data-rating="good"]').is_disabled(), 'corrupt storage cannot be overwritten by rating')
            page.evaluate('(key)=>localStorage.removeItem(key)', KEY)
            go('#topic=rl-parallel-planning')
            number = page.locator('[data-experiment=memory] input[type=number]').first
            number.fill('')
            check(page.locator('[data-experiment=memory] .experiment-output').inner_text() == '', 'invalid numeric input clears stale result')
            # Responsive reading and modal keyboard navigation.
            for width in (375, 768, 1440):
                page.set_viewport_size({'width': width, 'height': 900})
                for fragment, name in [('', 'home'), ('#view=questions', 'questions'), ('#topic=tp-matrix-communication', 'tp')]:
                    go(fragment)
                    check(page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), f'no viewport overflow {width}/{name}')
                    page.screenshot(path=str(args.output / f'{name}-{width}.png'), full_page=True)
                report['viewports'].append(width)
            page.set_viewport_size({'width': 375, 'height': 812})
            page.locator('#menu-toggle').click()
            check(page.locator('#mobile-directory').is_visible(), 'mobile directory opens')
            page.keyboard.press('Escape')
            check(not page.locator('#mobile-directory').is_visible(), 'mobile directory closes with Escape')
            check(page.locator('#menu-toggle').evaluate('(e)=>e===document.activeElement'), 'modal returns keyboard focus to opener')
            page.emulate_media(reduced_motion='reduce')
            go('#question=Q001')
            page.locator('.mobile-toc summary').click()
            page.locator('#mobile-toc a').first.click()
            check(page.locator('#article details').first.is_visible(), 'mobile TOC opens answer section')
            check(page.evaluate('(()=>{const ids=[...document.querySelectorAll("[id]")].map(e=>e.id);return new Set(ids).size===ids.length})()'), 'no duplicate DOM IDs')
            check(not errors, 'no uncaught JavaScript errors in full smoke run')
            report['status'] = 'passed'
        except Exception as error:
            report['status'] = 'failed'; report['error'] = str(error)
            page.screenshot(path=str(args.output / 'failure.png'), full_page=True)
            raise
        finally:
            report['runtime_errors'] = errors
            (args.output / 'browser-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            browser.close()
            if server: server.shutdown()
    print(json.dumps({k: report[k] for k in ('mode', 'status', 'routes', 'experiments', 'viewports')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
