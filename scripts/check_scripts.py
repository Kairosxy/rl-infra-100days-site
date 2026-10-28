"""Check external and inline browser JavaScript, including generated HTML."""
from __future__ import annotations
import argparse
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import tempfile

class Scripts(HTMLParser):
    def __init__(self):
        super().__init__(); self.current=None; self.scripts=[]
    def handle_starttag(self,tag,attrs):
        attributes=dict(attrs)
        if tag=='script' and not attributes.get('src') and attributes.get('type') not in ('application/json','application/ld+json'):
            self.current=[attributes.get('type')=='module',[]]
    def handle_endtag(self,tag):
        if tag=='script' and self.current is not None:
            self.scripts.append((self.current[0],''.join(self.current[1])));self.current=None
    def handle_data(self,text):
        if self.current is not None:self.current[1].append(text)

def check(root:Path):
    count=0
    with tempfile.TemporaryDirectory() as temp:
        for path in sorted(root.rglob('*')):
            if not path.is_file():continue
            if path.suffix=='.js':
                subprocess.run(['node','--check',str(path)],check=True);count+=1
            elif path.suffix=='.html':
                parser=Scripts();parser.feed(path.read_text(encoding='utf-8'))
                for i,(module,text) in enumerate(parser.scripts):
                    script=Path(temp)/('script.mjs' if module else 'script.js');script.write_text(text,encoding='utf-8')
                    result=subprocess.run(['node','--check',str(script)],capture_output=True,text=True)
                    if result.returncode:raise RuntimeError(f'{path} inline script {i}: {result.stderr}')
                    count+=1
    return count
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',nargs='?',type=Path,default=Path('dist'));args=parser.parse_args();print('Checked browser scripts:',check(args.root))
