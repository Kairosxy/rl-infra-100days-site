#!/usr/bin/env python3
"""Confirm that Pages serves this exact release, not merely a previous 200 page."""
from __future__ import annotations
import argparse
import json
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--sha', required=True)
    args = parser.parse_args()
    url = args.url.rstrip('/') + '/build-info.json?revision=' + args.sha
    last_error = ''
    for attempt in range(20):
        try:
            request = urllib.request.Request(url, headers={'Cache-Control': 'no-cache', 'User-Agent': 'RL-Infra-Release-Check'})
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.load(response)
            if payload.get('commit') != args.sha or payload.get('questions') != 224:
                raise ValueError('Live release identity or question count differs from expected artifact')
            print(json.dumps({'status': 'verified', 'url': args.url, **payload}))
            return
        except Exception as error:
            last_error = str(error)
            if attempt < 19:
                time.sleep(3)
    raise SystemExit('Deployment freshness check failed: ' + last_error)


if __name__ == '__main__':
    main()
