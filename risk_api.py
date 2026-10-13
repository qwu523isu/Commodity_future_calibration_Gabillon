"""Local, synchronous HTTP prototype. Standard library only; not a production server."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import re
import uuid
from Step4b_portfolio_risk_engine import BASE, main as run_portfolio


def create_server(port=8000, output_root=None):
    root = Path(output_root) if output_root is not None else BASE/'Output/API_Runs'

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload):
            data = json.dumps(payload, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                self.respond(200, {'status': 'ok', 'mode': 'synchronous-local-demo',
                                   'synthetic_volatility': True})
                return
            match = re.fullmatch(r'/runs/([0-9a-f]{32})', self.path)
            if match:
                result = root/match[1]/'portfolio_summary.json'
                if result.is_file():
                    self.respond(200, {'run_id': match[1], 'status': 'succeeded',
                                       'summary': json.loads(result.read_text(encoding='utf-8'))})
                    return
            self.respond(404, {'error': 'Run or endpoint not found.'})

        def do_POST(self):
            if self.path != '/runs':
                self.respond(404, {'error': 'Endpoint not found.'})
                return
            if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                self.respond(415, {'error': 'Content-Type must be application/json.'})
                return
            if self.headers.get('Transfer-Encoding'):
                self.respond(400, {'error': 'Chunked requests are unsupported.'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1000000:
                    self.respond(413, {'error': 'Require a JSON body of 1 to 1000000 bytes.'})
                    return
                config = json.loads(self.rfile.read(length))
                if not isinstance(config, dict):
                    raise ValueError('Configuration must be a JSON object.')
                n = config.get('scenario_count')
                if type(n) is not int or not 2 <= n <= 1000000:
                    raise ValueError('API scenario_count must be an integer from 2 to 1000000.')
            except (ValueError, UnicodeError) as error:
                self.respond(400, {'error': str(error)})
                return
            run_id = uuid.uuid4().hex
            directory = root/run_id
            try:
                directory.mkdir(parents=True)
                config_path = directory/'request_config.json'
                config_path.write_text(json.dumps(config), encoding='utf-8')
                summary = run_portfolio(config_path, directory)
            except (ValueError, KeyError, IndexError, TypeError) as error:
                self.respond(400, {'run_id': run_id, 'status': 'failed', 'error': str(error)})
                return
            except Exception:
                # Do not return internal filesystem details to clients.
                self.respond(500, {'run_id': run_id, 'status': 'failed', 'error': 'Risk calculation failed.'})
                return
            self.respond(201, {'run_id': run_id, 'status': 'succeeded', 'summary': summary})

    # Fixed loopback binding: no authentication or production serving layer yet.
    server = HTTPServer(('127.0.0.1', port), Handler)
    server.timeout = 1
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--output', type=Path, default=BASE/'Output/API_Runs')
    args = parser.parse_args()
    with create_server(args.port, args.output) as server:
        print(f'Local risk API: http://127.0.0.1:{server.server_port}; synthetic volatility demo.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
