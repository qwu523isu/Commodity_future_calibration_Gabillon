"""Financial invariants and a small CLI integration run; no extra test framework."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from Step4a_minimal_risk_engine import BASE, pnl_for_position
from Step4b_portfolio_risk_engine import CONFIG, load_config, risk


class RiskTests(unittest.TestCase):
    def test_position_direction_and_scaling(self):
        prices = np.array([70., 80., 90.])
        long = pnl_for_position(prices, 80., 1, 1000)
        np.testing.assert_array_equal(long, [-10000, 0, 10000])
        np.testing.assert_array_equal(long + pnl_for_position(prices, 80., -1, 1000), 0)
        np.testing.assert_array_equal(pnl_for_position(prices, 80., 2, 1000), 2*long)

    def test_known_tail_risk(self):
        # Worst 1.5 scenarios: (30 + 0.5*20)/1.5.
        var, es = risk(np.array([0., -10., -20., -30.]), 0.625)
        self.assertAlmostEqual(var, 18.75)
        self.assertAlmostEqual(es, 40/1.5)

    def test_invalid_config(self):
        original = load_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            for key, value in [('scenario_count', 101), ('confidence', 1), ('horizon_date', '2024-04-09')]:
                with self.subTest(key=key):
                    config = {**original, key: value}
                    path.write_text(json.dumps(config), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_cli_aggregation_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config()
            config['scenario_count'] = 1000
            path = root/'config.json'
            path.write_text(json.dumps(config), encoding='utf-8')
            for name in ('first', 'second'):
                result = subprocess.run([sys.executable, str(BASE/'run_risk_engine.py'),
                                         '--config', str(path), '--output', str(root/name)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            first, second = root/'first', root/'second'
            self.assertEqual((first/'portfolio_scenarios.csv').read_bytes(),
                             (second/'portfolio_scenarios.csv').read_bytes())
            frame = pd.read_csv(first/'portfolio_scenarios.csv')
            self.assertEqual(len(frame), 1000)
            legs = frame[['CLK2024_PnL_USD', 'CLM2024_PnL_USD']].sum(axis=1)
            np.testing.assert_allclose(frame.portfolio_PnL_USD, legs, atol=1e-9)
            summary = json.loads((first/'portfolio_summary.json').read_text())
            var, es = risk(frame.portfolio_PnL_USD.to_numpy(), config['confidence'])
            self.assertAlmostEqual(summary['portfolio_VaR_USD'], var)
            self.assertAlmostEqual(summary['portfolio_ES_USD'], es)
            self.assertEqual(json.loads((first/'run_config.json').read_text()), config)
            # An unknown contract must fail without publishing a result.
            config['positions'][0]['contract'] = 'UNKNOWN'
            path.write_text(json.dumps(config), encoding='utf-8')
            result = subprocess.run([sys.executable, str(BASE/'run_risk_engine.py'),
                                     '--config', str(path), '--output', str(root/'failed')],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('No market quote', result.stderr)
            self.assertFalse((root/'failed/portfolio_summary.json').exists())


if __name__ == '__main__':
    unittest.main()
