#!/usr/bin/env python3
"""Exercise generated pages over HTTP. Inline preview is not deployment QA."""
from __future__ import annotations
import argparse
import functools
import http.server
import json
import os
import re
import threading
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

KEY = 'rl-infra-100days-questionbank-v1'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dist',type=Path,default=Path('dist'))
    parser.add_argument('--output',type=Path,default=Path('test-artifacts'))
    parser.add_argument('--inline-preview',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    html=(args.dist/'index.html').read_text(encoding='utf-8')
    payload=json.loads(re.search(r'<script type="application/json" id="knowledge-data">(.*?)</script>',html,re.S).group(1))
    entries=[(kind,x['id'],x['title']) for kind,name in [('topic','topics'),('case','cases'),('implementation','implementations')] for x in payload[name]]
    questions=[('question',q['code'],q['question']) for q in payload['question_bank']]
    report={'mode':'inline-preview-with-storage-substitute' if args.inline_preview else 'http-real-localStorage','routes':0,'experiments':0,'viewports':[],'checks':[]}
    server=None
    if not args.inline_preview:
        handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(args.dist.resolve()))
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        base=f'http://127.0.0.1:{server.server_port}/'
    with sync_playwright() as p:
        executable=os.environ.get('CHROMIUM_EXECUTABLE')
        browser=p.chromium.launch(**({'executable_path':executable} if executable else {}))
        context=browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True)
        page=context.new_page()
        page.set_default_timeout(10000)
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.route('https://cdn.jsdelivr.net/**',lambda route:route.abort())
        if args.inline_preview:
            html=re.sub(r'<script[^>]+src=[^>]*>\s*</script>','',html)
            html=re.sub(r'<script type="module">.*?</script>','',html,flags=re.S)
            shim="<script>const store={};Object.defineProperty(window,'localStorage',{value:{getItem:k=>store[k]??null,setItem:(k,v)=>store[k]=String(v),removeItem:k=>delete store[k],clear:()=>Object.keys(store).forEach(k=>delete store[k])}});</script>"
            page.set_content(html.replace('<body>','<body>'+shim),wait_until='domcontentloaded')
        else:
            page.goto(base,wait_until='domcontentloaded')
        def go(fragment):
            page.evaluate('(h)=>{if(location.hash===h)dispatchEvent(new HashChangeEvent("hashchange"));else location.hash=h;}',fragment)
            page.wait_for_timeout(25)
        def check(condition,name):
            if not condition:
                raise AssertionError(name)
            report['checks'].append(name)
        try:
            check(page.locator('header').count()==1,'one global navigation header')
            check(len(questions)==224,'224 original questions in rendered product')
            for kind,ident,title in entries+questions:
                go(f'#{kind}={ident}')
                expect(page.locator('#detail-title')).to_have_text(title)
                check(page.locator('#detail-title').inner_text()==title,f'route {kind}:{ident}')
                check(page.locator('#detail').is_visible(),f'visible {kind}:{ident}')
                report['routes']+=1
            check(not errors,'all content routes without runtime exceptions')
            for spec in payload['learning_map']['experiments']:
                kind,ident=spec['entry'].split(':',1)
                go(f'#{kind}={ident}')
                widget=page.locator(f'[data-experiment="{spec["type"]}"]')
                expect(widget).to_have_count(1)
                check(widget.count()==1,'native experiment '+spec['type'])
                check(widget.locator('.experiment-output').inner_text().strip()!='','nonempty model '+spec['type'])
                select=widget.locator('select').first
                if select.count():
                    options=select.locator('option').evaluate_all('(os)=>os.map(o=>o.value)')
                    select.select_option(options[-1])
                    check(widget.locator('.experiment-output').inner_text().strip()!='','updated model '+spec['type'])
                report['experiments']+=1
            check(page.locator('iframe').count()==0,'interactions belong to article, not embedded web pages')
            for fragment in ['#topic=tp-matrix-communication','#topic=pp-pipeline-bubble','#question=Q081','#topic=tp-matrix-communication']:
                go(fragment)
            check(not errors,'interactive article lifecycle regression')
            go('')
            expect(page.locator('#learning-paths')).to_be_visible()
            check(page.locator('#learning-paths .route-row').count() == len(payload['learning_map']['paths']),
                  'home displays all curated learning paths')
            check(page.locator('#knowledge-map .map-node').count() == len(payload['chapters']) == 11,
                  'eleven distinct domains on the knowledge map')
            check(page.locator('#knowledge-map .map-edge').count() == len(payload['learning_map']['domain_map']['edges']),
                  'knowledge map renders each declared relationship')
            check(page.locator('#knowledge-map .map-node[data-area="serving"]').get_attribute('href') == '#area=serving',
                  'map node routes to unified scope')
            page.locator('#knowledge-map .map-node[data-area="parallel"]').click()
            expect(page.locator('#scope-name')).to_have_text(next(c['title'] for c in payload['chapters'] if c['id']=='parallel'))
            check(page.locator('#collection [data-group]').count() == 3,
                  'map node opens all three content types')
            page.go_back()
            expect(page.locator('#learning-paths')).to_be_visible()
            page.locator('#learning-paths .route-steps a').first.click()
            first_topic = payload['learning_map']['paths'][0]['topics'][0]
            expect(page.locator('#detail-title')).to_have_text(next(t['title'] for t in payload['topics'] if t['id'] == first_topic))
            page.go_back()
            expect(page.locator('#knowledge-map')).to_be_visible()
            check(page.locator('#chapter-list').count() == 1, 'one directory DOM')
            branch = page.locator('.directory-node[data-depth="0"]').filter(
                has=page.locator('.scope-link[data-scope="parallel||||"]'))
            toggle = branch.locator(':scope > .directory-line > .directory-toggle')
            if toggle.get_attribute('aria-expanded') != 'true':
                toggle.click()
            check(toggle.get_attribute('aria-expanded') == 'true', 'chapter expands independently of selection')
            go('#view=paths')
            expect(page.locator('#learning-paths')).to_be_visible()
            check(page.locator('#learning-paths .route-steps a').count() > 0, 'legacy paths link resolves to the same homepage')
            go('#view=questions&kind=case')
            check(page.locator('#collection .entry').count() > 0, 'legacy engineering Issues entry')
            check(page.locator('#content-tabs [aria-current=page]').get_attribute('data-mode') == 'issues',
                  'legacy Issues route selects the shared tab')
            go('#topic=tp-matrix-communication')
            check(page.locator('#article img').first.evaluate('(e)=>e.complete && e.naturalWidth>0'),'TP static figure loads')
            for topic_id in ('pp-pipeline-bubble', 'ep-token-routing', 'kv-scheduling', 'fp8-numerics', 'ppo-grpo', 'async-rollout'):
                go('#topic=' + topic_id)
                check(page.locator('#article img').count() > 0, topic_id + ' has static figure')
                check(page.locator('#article img').evaluate_all('(imgs)=>imgs.every(e=>e.complete && e.naturalWidth>0)'), topic_id + ' figures load')
            go('#topic=tp-matrix-communication')
            scope_link = page.locator('#chapter-list .scope-link[data-ref="tp-matrix-communication"]')
            check(scope_link.get_attribute('aria-current') == 'page', 'direct article resolves to topic scope')
            for mode in ('issues', 'implementation', 'topic', 'all'):
                page.locator(f'#content-tabs [data-mode={mode}]').click()
                expect(page.locator('#collection')).to_be_visible()
                expect(page.locator('#content-tabs [aria-current=page]')).to_have_attribute('data-mode', mode)
                check('ref=tp-matrix-communication' in page.url, f'{mode} preserves topic scope')
                check(scope_link.get_attribute('aria-current') == 'page', f'{mode} preserves selected directory node')
                check(page.locator('#content-tabs [aria-current=page]').get_attribute('data-mode') == mode,
                      f'{mode} uses the same content tab strip')
                check(page.locator('#collection .entry').count() > 0, f'{mode} has related content')
            check(page.locator('#collection [data-group]').count() == 3,
                  'topic overview integrates knowledge, Issues and implementation')
            page.locator('#content-tabs [data-mode=issues]').click()
            # Change a directory scope while staying in Issues. A folder click must not return to Knowledge.
            page.locator('#chapter-list .scope-link[data-scope="parallel|parallel-2|||"]').click()
            expect(page.locator('#scope-name')).to_have_text('张量与流水线并行')
            check(page.locator('#content-tabs [aria-current=page]').get_attribute('data-mode') == 'issues',
                  'directory selection preserves Issues mode')
            check('section=parallel-2' in page.url and 'ref=' not in page.url, 'section selection widens topic scope')
            page.locator('#search').fill('Q086')
            expect(page.locator('#collection .entry')).to_have_count(1)
            check('section=parallel-2' in page.url and 'q=Q086' in page.url, 'search retains directory scope and type')
            search_url = page.url
            page.locator('#collection .entry').click()
            expect(page.locator('#detail-title')).to_have_text(questions[85][2])
            check(page.locator('#search').input_value() == '', 'opening result exposes related knowledge without search filtering')
            check(int(page.locator('#content-tabs [data-mode=topic] .tab-count').inner_text()) > 0,
                  'search result detail exposes related knowledge')
            page.locator('#breadcrumbs .back').click()
            expect(page.locator('#collection')).to_be_visible()
            expect(page.locator('#search')).to_have_value('Q086')
            expect(page.locator('#collection .entry')).to_have_count(1)
            check(page.url == search_url and page.locator('#search').input_value() == 'Q086', 'return restores precise search state')
            page.locator('#collection .entry').click()
            expect(page.locator('#detail')).to_be_visible()
            page.go_back()
            expect(page.locator('#collection')).to_be_visible()
            expect(page.locator('#search')).to_have_value('Q086')
            expect(page.locator('#collection .entry')).to_have_count(1)
            check(page.url == search_url, 'browser Back restores the same filtered list')
            go('#view=questions&kind=question')
            check('224' in page.locator('#collection .count').inner_text(),'question filter includes all 224')
            page.locator('#search').fill('Q081')
            expect(page.locator('#collection .entry')).to_have_count(1)
            check(page.locator('#collection .entry').count()==1,'search stays within question scope')
            page.locator('#collection .entry').click()
            expect(page.locator('#detail-title')).to_have_text(questions[80][2])
            check(page.locator('#detail-title').inner_text()==questions[80][2],'search result deep link')
            go('#view=questions&area=parallel&section=parallel-2')
            check(page.locator('#collection .entry').count()>0,'section-scoped legacy questions are not empty')
            go('#area=frameworks&impl=megatron-parallel-groups')
            check(page.locator('#collection [data-group]').count() == 3, 'framework scope shows related knowledge, issues and implementations')
            page.locator('#content-tabs [data-mode=implementation]').click()
            expect(page.locator('#content-tabs [aria-current=page]')).to_have_attribute('data-mode', 'implementation')
            check(page.locator('#collection .entry').count() == 1, 'framework scope does not leak unrelated implementations')
            check(page.locator('#collection .entry').get_attribute('data-entry') == 'implementation:megatron-parallel-groups',
                  'exact framework implementation selected')
            original={'version':1,'records':{'81':{'answer':'saved-draft','layers':{'c1':'saved-layer'},'note':'saved-note','rating':'hard','attempts':1,'streak':0,'due':'2026-01-01','bookmarked':True,'history':[{'rating':'hard','date':'2026-01-01T12:00:00Z'}]},'2':{'answer':'other-record'}}}
            page.evaluate('([key,value])=>localStorage.setItem(key,JSON.stringify(value))',[KEY,original])
            go('#question=Q081')
            page.locator('[data-rating="good"]').click()
            stored=page.evaluate('(key)=>JSON.parse(localStorage.getItem(key))',KEY)
            check(stored['records']['81']['answer']=='saved-draft' and stored['records']['81']['layers']['c1']=='saved-layer','rating preserves legacy draft and layers')
            check(stored['records']['2']['answer']=='other-record','rating preserves other questions')
            check(stored['records']['81']['attempts']==2 and stored['records']['81']['rating']=='good','rating writes same legacy storage schema')
            go('#view=questions&review=bookmarked')
            check(page.locator('#collection .entry').count()==1,'bookmark filter')
            incoming={'format':'rl-infra-interview-practice-v1','records':{'81':{'note':'must-not-overwrite'},'3':{'note':'imported'}}}
            page.locator('#collection input[type=file]').set_input_files({'name':'records.json','mimeType':'application/json','buffer':json.dumps(incoming).encode()})
            page.wait_for_function('(key)=>!!JSON.parse(localStorage.getItem(key)).records[3]',arg=KEY)
            stored=page.evaluate('(key)=>JSON.parse(localStorage.getItem(key))',KEY)
            check(stored['records']['81']['note']=='saved-note' and stored['records']['3']['note']=='imported','import merges without overwriting existing records')
            incoming['records']['999']={'rating':'easy'}
            page.locator('#collection input[type=file]').set_input_files({'name':'invalid.json','mimeType':'application/json','buffer':json.dumps(incoming).encode()})
            page.wait_for_timeout(80)
            check(page.evaluate('(key)=>JSON.parse(localStorage.getItem(key))',KEY)==stored,'invalid import is atomic and does not change stored data')
            page.locator('.review-panel summary').click()
            with page.expect_download() as download_info:
                page.locator('#collection .review-tools button').first.click()
            download_info.value.save_as(args.output/'review-roundtrip.json')
            backup=json.loads((args.output/'review-roundtrip.json').read_text())
            check(backup['records']==stored['records'],'export preserves records for old quiz import')
            if not args.inline_preview:
                page.reload(wait_until='domcontentloaded')
                check(page.evaluate('(key)=>JSON.parse(localStorage.getItem(key))',KEY)==stored,'real localStorage persists across reload')
                page.goto(base+'quiz/#q=81',wait_until='domcontentloaded')
                expect(page.locator('#detail-title')).to_be_visible()
                check('question=Q081' in page.url,'legacy quiz bookmark forwards to reader')
                check(page.evaluate('(key)=>JSON.parse(localStorage.getItem(key))',KEY)==stored,'legacy quiz redirect preserves storage')
                page.goto(base,wait_until='domcontentloaded')
            page.evaluate('(key)=>localStorage.setItem(key,"not-valid-json")',KEY)
            go('#question=Q081')
            check(page.locator('[data-rating="good"]').is_disabled(),'corrupt storage cannot be overwritten by rating')
            page.evaluate('(key)=>localStorage.removeItem(key)',KEY)
            go('#topic=rl-parallel-planning')
            page.locator('[data-experiment=memory] input[type=number]').first.fill('')
            check(page.locator('[data-experiment=memory] .experiment-output').inner_text()=='','invalid numeric input clears stale result')
            for width in (375,768,1440):
                page.set_viewport_size({'width':width,'height':900})
                for fragment,name in [('', 'home'),('#view=questions','questions'),('#area=parallel&section=parallel-2','scope'),('#topic=tp-matrix-communication','tp')]:
                    go(fragment)
                    check(page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'),f'no viewport overflow {width}/{name}')
                    page.screenshot(path=str(args.output/f'{name}-{width}.png'),full_page=True)
                report['viewports'].append(width)
            page.set_viewport_size({'width':375,'height':812})
            page.locator('#menu-toggle').click()
            check(page.locator('#mobile-directory').is_visible(),'mobile directory opens')
            check(page.locator('#mobile-tree #chapter-list').count() == 1, 'mobile drawer uses the same directory DOM')
            page.keyboard.press('Escape')
            check(not page.locator('#mobile-directory').is_visible(),'mobile directory closes with Escape')
            check(page.locator('#menu-toggle').evaluate('(e)=>e===document.activeElement'),'modal returns keyboard focus to opener')
            go('#mode=issues&area=parallel')
            page.locator('#menu-toggle').click()
            page.locator('#mobile-tree .scope-link[data-scope="parallel|parallel-2|||"]').click()
            expect(page.locator('#scope-name')).to_have_text('张量与流水线并行')
            expect(page.locator('#collection')).to_be_visible()
            check(not page.locator('#mobile-directory').is_visible(), 'mobile scope selection closes drawer')
            check('section=parallel-2' in page.url and page.locator('#content-tabs [aria-current=page]').get_attribute('data-mode') == 'issues', 'mobile directory keeps content mode')
            page.emulate_media(reduced_motion='reduce')
            go('#question=Q001')
            page.locator('.mobile-toc summary').click()
            page.locator('#mobile-toc a').first.click()
            check(page.locator('#article details').first.is_visible(),'mobile TOC opens answer section')
            check(page.evaluate('(()=>{const ids=[...document.querySelectorAll("[id]")].map(e=>e.id);return new Set(ids).size===ids.length})()'),'no duplicate DOM IDs')
            check(not errors,'no uncaught JavaScript errors in full smoke run')
            report['status']='passed'
        except Exception as error:
            report['status']='failed'
            report['error']=str(error)
            report['location']=page.url
            report['detail_title']=page.locator('#detail-title').inner_text()
            page.screenshot(path=str(args.output/'failure.png'),full_page=True)
            raise
        finally:
            report['runtime_errors']=errors
            (args.output/'browser-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            browser.close()
            if server:
                server.shutdown()
    print(json.dumps({k:report[k] for k in ('mode','status','routes','experiments','viewports')},ensure_ascii=False))

if __name__=='__main__':
    main()
