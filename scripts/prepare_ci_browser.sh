#!/usr/bin/env bash
# GitHub's Ubuntu runner already supplies most browser libraries.
# Retain the pinned Playwright browser; install OS packages only when needed.
set -euo pipefail

python -m pip install playwright==1.57.0
python -m playwright install --only-shell chromium

check_browser() {
    python - <<'PY'
from playwright.sync_api import sync_playwright

with sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    page = browser.new_page()
    page.set_content("<p>RL Infra</p>")
    assert page.locator("p").inner_text() == "RL Infra"
    browser.close()
PY
}

if ! check_browser; then
    echo "::warning::Pinned Chromium launch failed; installing required system libraries"
    python -m playwright install-deps chromium
    check_browser
fi

# Chinese text must have a real font in responsive screenshots.
if [[ -z "$(fc-list :lang=zh family)" ]]; then
    echo "::warning::No Chinese font found; installing only the missing font"
    sudo apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 install -y --no-install-recommends fonts-wqy-zenhei
fi
if [[ -z "$(fc-list :lang=zh family)" ]]; then
    echo "::error::Chinese font coverage is missing"
    exit 1
fi
echo "Chromium launch and Chinese font coverage verified."
