"""Validate optional product metadata and derive read-only question links."""
from __future__ import annotations
import json
from pathlib import Path

EXPERIMENTS = {'tp', 'pp', 'ep', 'topology', 'cost', 'memory', 'context', 'sharding', 'collective', 'policy', 'baseline', 'mask', 'ppo', 'grpo'}


def enrich_payload(root: Path, catalog: dict, bank: dict) -> dict:
    meta = json.loads((root / 'knowledge' / 'learning-map.json').read_text(encoding='utf-8'))
    if meta.get('schema') != 'rl-infra-learning-map-v1':
        raise ValueError('Invalid learning-map schema')
    topics = {x['id'] for x in catalog['topics']}
    entries = {f'{kind}:{x["id"]}' for kind, group in [('topic','topics'),('case','cases'),('implementation','implementations')] for x in catalog[group]}
    def require(ids):
        if not isinstance(ids, list) or any(x not in topics for x in ids) or len(ids) != len(set(ids)):
            raise ValueError(f'Invalid topic references: {ids}')
    for path in meta.get('paths', []):
        require(path['topics'])
        if not path.get('title') or not path.get('goal'):
            raise ValueError('Reading path must describe an actual goal')
    for stage in meta.get('system_flow', []):
        require([stage['topic']])
    prereq = meta.get('prerequisites', {})
    for topic, ids in prereq.items():
        require([topic])
        require(ids)
    visiting, visited = set(), set()
    def visit(topic):
        if topic in visiting:
            raise ValueError(f'Cyclic prerequisite: {topic}')
        if topic in visited:
            return
        visiting.add(topic)
        for predecessor in prereq.get(topic, []):
            visit(predecessor)
        visiting.remove(topic)
        visited.add(topic)
    for topic in topics:
        visit(topic)
    for experiment in meta.get('experiments', []):
        if experiment.get('entry') not in entries or experiment.get('type') not in EXPERIMENTS or not experiment.get('after'):
            raise ValueError(f'Invalid embedded experiment: {experiment}')
    by_code = {q['code']:q for q in bank['questions']}
    for code, ids in meta.get('question_links', {}).items():
        if code not in by_code:
            raise ValueError(f'Unknown question: {code}')
        require(ids)
        if not ids:
            raise ValueError(f'Question without a topic: {code}')
    questions = []
    for q in bank['questions']:
        # Preserve all original fields; the additional links do not rewrite source material.
        questions.append({**q, 'source_topic': q['topic'], 'topics': meta.get('question_links', {}).get(q['code'], [q['topic']])})
    catalog['question_bank'] = questions
    catalog['question_note'] = bank.get('content_note', '')
    catalog['question_sources'] = bank.get('sources', [])
    catalog['learning_map'] = meta
    return catalog
