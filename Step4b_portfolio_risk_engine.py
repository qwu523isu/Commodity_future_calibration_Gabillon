"""Minimal shared-factor portfolio demo; synthetic volatility, no market forecast."""
import json
import numpy as np
import pandas as pd
from Step4a_minimal_risk_engine import BASE, integrated_decay, pnl_for_position

OUT = BASE / 'Output/Step4b_Portfolio_Risk_Engine'
CONFIG = BASE / 'Input_data/risk_engine_config.json'


def load_config(config_path=CONFIG):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    # The demo calendar has only this reviewed next-trading-day pair.
    if (config['valuation_date'], config['horizon_date']) != ('2024-04-05', '2024-04-08'):
        raise ValueError('Only 2024-04-05 to 2024-04-08 is supported until the trading calendar is added.')
    n, seed, confidence = (config[k] for k in ('scenario_count', 'seed', 'confidence'))
    if type(n) is not int or n < 2 or n % 2:
        raise ValueError('scenario_count must be a positive even integer >= 2.')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    if not 0 < confidence < 1 or (1-confidence)*n < 1:
        raise ValueError('confidence must be between 0 and 1, with at least one tail scenario.')
    vol = config['synthetic_volatility']
    if not all(np.isfinite(vol[k]) for k in ('sigma_X', 'sigma_Y', 'rho')):
        raise ValueError('Volatility inputs must be finite.')
    if vol['sigma_X'] <= 0 or vol['sigma_Y'] <= 0 or not -1 < vol['rho'] < 1:
        raise ValueError('Require positive volatilities and -1 < rho < 1.')
    positions = config['positions']
    if not positions or len({p['contract'] for p in positions}) != len(positions):
        raise ValueError('Provide a nonempty position list with one net quantity per contract.')
    for p in positions:
        if not np.isfinite(p['quantity']) or not np.isfinite(p['multiplier']) or p['multiplier'] <= 0:
            raise ValueError('Require finite quantities and positive finite multipliers.')
    return config


def risk(pnl, confidence=0.95):
    losses = -np.asarray(pnl)
    mass = (1-confidence)*len(losses)
    count = int(np.floor(mass))
    ordered = np.sort(losses)[::-1]
    es = (ordered[:count].sum() + (mass-count)*ordered[count]) / mass
    return float(np.quantile(losses, confidence)), float(es)


def main(config_path=CONFIG, output_dir=OUT):
    config = load_config(config_path)
    date, horizon_date = config['valuation_date'], config['horizon_date']
    confidence = config['confidence']
    market = pd.read_csv(BASE/'Output/Step1a_EIA_Preparation/EIA_WTI_calibration_input.csv')
    curves = pd.read_csv(BASE/'Output/Step1c_EIA_Full_Sample/EIA_calibration_results.csv')
    curve = curves.loc[curves.Date == date].iloc[0]
    market = market.loc[market.Date == date].set_index('Contract')
    if not market.index.is_unique:
        raise ValueError('Duplicate market quotes for a contract.')
    kx, ky = float(curve.kappa_X), float(curve.kappa_Y)
    sx, sy, rho = (config['synthetic_volatility'][k] for k in ('sigma_X', 'sigma_Y', 'rho'))
    h = (pd.Timestamp(horizon_date)-pd.Timestamp(date)).days / 365
    n, seed = config['scenario_count'], config['seed']
    # Exact two-factor integrated innovation covariance, shared by all contracts.
    cross = rho*sx*sy*integrated_decay(kx+ky, 0, h)
    covariance = np.array([[sx*sx*integrated_decay(2*kx, 0, h), cross],
                           [cross, sy*sy*integrated_decay(2*ky, 0, h)]])
    rng = np.random.default_rng(seed)
    half = rng.standard_normal((n//2, 2))
    factors = np.vstack([half, -half]) @ np.linalg.cholesky(covariance).T
    scenario_table = pd.DataFrame({'scenario': np.arange(n)})
    pnls, details = [], []
    for position in config['positions']:
        name, quantity, multiplier = (position[k] for k in ('contract', 'quantity', 'multiplier'))
        if name not in market.index:
            raise ValueError(f'No market quote for {name} on {date}.')
        row = market.loc[name]
        maturity, initial = float(row.Maturity), float(row.Price)
        if not 0 < h < maturity:
            raise ValueError(f'{name} expires within the risk horizon.')
        loading = np.exp(-np.array([kx, ky])*(maturity-h))
        variance = float(loading @ covariance @ loading)
        expected_variance = (sx*sx*integrated_decay(2*kx, maturity-h, maturity)
                             + sy*sy*integrated_decay(2*ky, maturity-h, maturity)
                             + 2*rho*sx*sy*integrated_decay(kx+ky, maturity-h, maturity))
        np.testing.assert_allclose(variance, expected_variance, rtol=1e-12)
        prices = initial*np.exp(factors @ loading - 0.5*variance)
        pnl = pnl_for_position(prices, initial, quantity, multiplier)
        unit_pnl = pnl_for_position(prices, initial, 1, multiplier)
        np.testing.assert_allclose(unit_pnl + pnl_for_position(prices, initial, -1, multiplier), 0)
        np.testing.assert_allclose(pnl_for_position(prices, initial, 2, multiplier), 2*unit_pnl)
        assert pnl_for_position(initial, initial, quantity, multiplier) == 0
        var, es = risk(pnl, confidence)
        details.append({**position, 'price': initial, 'maturity': maturity,
                        'standalone_VaR_USD': var, 'standalone_ES_USD': es})
        scenario_table[f'{name}_price'] = prices
        scenario_table[f'{name}_PnL_USD'] = pnl
        pnls.append(pnl)
    portfolio = np.sum(pnls, axis=0)
    np.testing.assert_allclose(portfolio, scenario_table.filter(regex='_PnL_USD$').sum(axis=1))
    assert np.isfinite(portfolio).all()
    var, es = risk(portfolio, confidence)
    assert es >= var
    scenario_table['portfolio_PnL_USD'] = portfolio
    scenario_table['portfolio_loss_USD'] = -portfolio
    summary = {
        'purpose': 'ARCHITECTURE VALIDATION ONLY; SYNTHETIC VOLATILITY',
        'valuation_date': date, 'horizon_date': horizon_date,
        'horizon_business_days': 1, 'horizon_calendar_days': 3, 'year_fraction': 'ACT/365F',
        'positions': details, 'scenario_count': n, 'seed': seed, 'confidence': confidence,
        'portfolio_VaR_USD': var, 'portfolio_ES_USD': es,
        'mean_portfolio_PnL_USD': float(portfolio.mean()),
        'sigma_X_synthetic': sx, 'sigma_Y_synthetic': sy, 'rho_synthetic': rho,
        'kappa_X': kx, 'kappa_Y': ky,
        'curve_converged': bool(curve.converged), 'curve_any_bound': bool(curve.any_bound),
        'drift_assumption': 'zero expected futures price return',
        'checks': 'integrated covariance, zero price change, offsetting positions, quantity scaling, aggregation, finite values, ES >= VaR passed',
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    scenario_table.to_csv(output_dir/'portfolio_scenarios.csv', index=False)
    (output_dir/'portfolio_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report = (
        '组合风险引擎：仅用于架构验证，波动率为合成假设\n'
        f'估值日 {date}；风险日 {horizon_date}（下一交易日，跨周末3日）。\n'
        + ''.join(f"头寸：{p['contract']}，数量 {p['quantity']} 手，乘数 {p['multiplier']}。\n" for p in config['positions'])
        +
        '不同月份共享同一组双因子情景，分别重估后逐情景汇总盈亏。\n'
        f'情景数 {n:,}；组合{confidence:.0%} VaR {var:,.2f} 美元；ES {es:,.2f} 美元。\n'
        '组合风险直接由组合损失分布计算，不能将单个头寸VaR相加。\n'
        '检查通过：协方差公式、零价格变化、同合约反向抵消、数量缩放、盈亏汇总、有限值、ES不小于VaR。\n'
        f'校准收敛 {bool(curve.converged)}；存在触界 {bool(curve.any_bound)}。\n'
        f'合成波动率 {sx}/{sy}，相关性 {rho}；假设期货价格期望不变。\n'
        'ES为最差尾部的平均损失；尾部数量不是整数时使用分数权重。\n'
        '仅支持上述固定估值日和风险日；尚未包括通用交易日历、费用、流动性或交割。\n'
        '新增第二个因子随机数后，随机数映射与单头寸文件不同，蒙特卡洛数字不要求逐项一致。\n'
    )
    (output_dir/'portfolio_report.txt').write_text(report, encoding='utf-8')
    print(f'Portfolio demo complete: {confidence:.0%} VaR USD {var:,.2f}; ES USD {es:,.2f}. Checks passed.')
    return summary


if __name__ == '__main__':
    main()
