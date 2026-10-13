"""Check a running API, then optionally verify saved results after restart."""
import argparse
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler


def request(base, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(base+path, data=body, headers={'Content-Type': 'application/json'})
    with build_opener(ProxyHandler({})).open(req, timeout=30) as response:
        return response.status, json.loads(response.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--state-file', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    for attempt in range(60):
        try:
            status, health = request(args.url, '/health')
            if status != 200 or health.get('status') != 'ok' or not health.get('synthetic_volatility'):
                raise RuntimeError('Unexpected health response.')
            break
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 59:
                raise RuntimeError('API did not become ready within the startup window.')
            time.sleep(1)
    if args.verify_only:
        original = json.loads(args.state_file.read_text(encoding='utf-8'))
    else:
        config = json.loads((Path(__file__).resolve().parents[1]/'Input_data/risk_engine_config.json').read_text())
        config['scenario_count'] = 1000
        status, original = request(args.url, '/runs', config)
        if status != 201 or original.get('status') != 'succeeded':
            raise RuntimeError('Risk submission failed.')
        summary = original['summary']
        var, es = summary['portfolio_VaR_USD'], summary['portfolio_ES_USD']
        if not math.isfinite(var) or not math.isfinite(es) or es < var or summary['scenario_count'] != 1000:
            raise RuntimeError('Invalid risk result.')
        args.state_file.write_text(json.dumps(original), encoding='utf-8')
    status, saved = request(args.url, '/runs/'+original['run_id'])
    if status != 200 or saved != original:
        raise RuntimeError('Saved result does not match the submitted result.')
    try:
        request(args.url, '/runs/'+'0'*32)
    except HTTPError as error:
        if error.code != 404:
            raise
    else:
        raise RuntimeError('Unknown run must return 404.')
    print('API smoke check passed' + (' after restart.' if args.verify_only else '.'))


if __name__ == '__main__':
    main()
