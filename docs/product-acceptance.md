# Knowledge product acceptance contract

## Source of truth

- `knowledge/catalog.json`: areas, sections, ordered documents, implementation and case relations.
- `knowledge/learning-map.json`: system flow, reading paths, prerequisite DAG, native experiment placement, reviewed question-topic overrides.
- `quiz/data/questions.json`: unchanged original 224 questions. Original source wording, answers and alternative titles are retained. Missing original equations and unverified claims are disclosed, not silently invented.
- `knowledge/site/core.js`: pure models and compatible local review state; `experiments.js`: native document figures; `app.js`: routing and reader.

## Required release gates

```
python scripts/build_knowledge.py --output dist/index.html
python scripts/build_quiz.py --output dist/quiz/index.html
python -m unittest discover -s scripts -p 'test_*.py' -v
node scripts/test_mechanisms.js
python scripts/check_scripts.py dist
python -m pip install playwright==1.57.0
python -m playwright install --with-deps chromium
python scripts/browser_smoke.py --dist dist --output test-artifacts
```

Browser checks visit every document and Q001-Q224, switch between interactive documents, operate every native experiment, validate scoped search and local review import/export, preserve old drafts, reject corrupt state, and check 375/768/1440px layouts. Screenshots and a machine-readable report are uploaded by CI. An inline-preview run uses a storage substitute and must not be reported as HTTP or deployment verification.

Successful upload is not successful deployment. The deploy job checks that live `build-info.json` has the exact `GITHUB_SHA` and 224 questions. A later failure does not authorize calling an older published page the new release.

## Product behavior

One header; hierarchical directory; problem and document search share URL state; question IDs remain stable. Native experiments are placed in relevant article sections, not nested full-page iframes. Old lab URLs forward to their articles. Markdown source links remain traceable. Mermaid and MathJax are optional pinned CDN renderers; the reader and all native figures work if their network requests fail. Diagram source remains accessible.

Review data uses the original browser-local key and numeric question IDs. Every write re-reads storage; imports validate completely before merging, keep existing records, and report conflicts. No server uploads, analytics, fabricated study scores, or assumed cross-domain synchronization.

## Deliberate model boundaries

TP distinguishes single-batch latency, fixed-GPU proxy throughput, compute and weight-read HBM lower bounds. PP diagrams implement actual forward/backward dependencies for GPipe and 1F1B; tick durations are assumptions. EP always has eight experts and distinct top-2 assignments; only placement changes. KV budget includes TP head and PP layer sharding and separates Learner/Rollout settings. None is a measured performance predictor or a validated production launcher.

This release does not certify every original mathematical answer, real-GPU performance, all framework API versions, or physical iPhone Safari behavior. These require separate evidence. Chromium viewport tests must not be described as iPhone hardware tests.
