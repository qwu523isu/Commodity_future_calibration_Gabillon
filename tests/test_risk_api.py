import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler
from risk_api import create_server
from Step4b_portfolio_risk_engine import load_config


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.server = create_server(0, self.directory.name)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.client = build_opener(ProxyHandler({}))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.directory.cleanup()

    def request(self, path, config=None):
        data = None if config is None else json.dumps(config).encode()
        request = Request(self.base+path, data=data,
                          headers={'Content-Type': 'application/json'})
        try:
            response = self.client.open(request, timeout=30)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read())

    def test_run_and_query(self):
        status, health = self.request('/health')
        self.assertEqual(status, 200)
        self.assertTrue(health['synthetic_volatility'])
        config = load_config()
        config['scenario_count'] = 1000
        status, result = self.request('/runs', config)
        self.assertEqual(status, 201)
        self.assertEqual(result['status'], 'succeeded')
        summary = result['summary']
        self.assertGreaterEqual(summary['portfolio_ES_USD'], summary['portfolio_VaR_USD'])
        status, saved = self.request('/runs/'+result['run_id'])
        self.assertEqual(status, 200)
        self.assertEqual(saved, result)
        self.assertTrue((Path(self.directory.name)/result['run_id']/'portfolio_scenarios.csv').is_file())

    def test_reject_invalid_requests(self):
        status, _ = self.request('/runs/'+'0'*32)
        self.assertEqual(status, 404)
        config = load_config()
        config['scenario_count'] = 1001
        status, result = self.request('/runs', config)
        self.assertEqual(status, 400)
        self.assertEqual(result['status'], 'failed')
        status, _ = self.request('/runs/'+result['run_id'])
        self.assertEqual(status, 404)
        self.assertFalse((Path(self.directory.name)/result['run_id']/'portfolio_summary.json').exists())
        status, _ = self.request('/runs', [])
        self.assertEqual(status, 400)


if __name__ == '__main__':
    unittest.main()
