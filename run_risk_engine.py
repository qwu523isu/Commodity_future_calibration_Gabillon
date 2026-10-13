"""Single batch entry point. Returns a nonzero exit code on failure."""
import argparse
from pathlib import Path
import sys
from Step4b_portfolio_risk_engine import CONFIG, OUT, main as run_portfolio


def main():
    parser = argparse.ArgumentParser(description='Run the synthetic-volatility risk architecture demo.')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    try:
        run_portfolio(args.config.resolve(), args.output.resolve())
    except (ValueError, KeyError, IndexError, OSError, TypeError) as error:
        print(f'Risk run failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
