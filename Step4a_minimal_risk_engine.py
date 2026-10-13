"""Architecture demo: observed futures -> factor shocks -> P&L -> VaR/ES.

Synthetic volatility and zero expected futures return are explicit assumptions.
This is not the original spot simulation or a calibrated physical-measure model.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / 'Output' / 'Step4a_Minimal_Risk_Engine'


def integrated_decay(rate, start, end):
    if rate == 0:
        return end - start
    return np.exp(-rate * start) * (-np.expm1(-rate * (end - start))) / rate


def pnl_for_position(prices, initial_price, contracts, multiplier):
    return (prices - initial_price) * contracts * multiplier


def main():
    market = pd.read_csv(BASE / 'Output/Step1a_EIA_Preparation/EIA_WTI_calibration_input.csv')
    curves = pd.read_csv(BASE / 'Output/Step1c_EIA_Full_Sample/EIA_calibration_results.csv')
    date = market.Date.max()
    position = market.loc[(market.Date == date) & (market.NearbyRank == 1)].iloc[0]
    curve = curves.loc[curves.Date == date].iloc[0]
    # Illustrative long one standard CL contract: 1,000 barrels.
    contracts, multiplier = 1, 1000
    sigma_x, sigma_y, rho = 0.35, 0.20, 0.30
    scenarios, seed, confidence = 100000, 42, 0.95
    # One business-day advance; weekend calendar time is included. Holiday
    # handling is not provided by BDay, so restrict this demo to the saved date.
    if date != '2024-04-05':
        raise ValueError('Review the next trading date before extending this demo.')
    horizon_date = pd.Timestamp('2024-04-08')
    horizon = (horizon_date - pd.Timestamp(date)).days / 365.0
    maturity = float(position.Maturity)
    if not 0 < horizon < maturity:
        raise ValueError('Position must remain unexpired through the risk horizon.')
    kx, ky = float(curve.kappa_X), float(curve.kappa_Y)
    # Exact integrated loadings over the horizon for the fixed-expiry future.
    start, end = maturity - horizon, maturity
    variance = (sigma_x**2 * integrated_decay(2*kx, start, end)
                + sigma_y**2 * integrated_decay(2*ky, start, end)
                + 2*rho*sigma_x*sigma_y*integrated_decay(kx+ky, start, end))
    if variance < 0 or not np.isfinite(variance):
        raise ValueError('Invalid futures return variance.')
    initial = float(position.Price)
    rng = np.random.default_rng(seed)
    z_half = rng.standard_normal(scenarios // 2)
    z = np.concatenate([z_half, -z_half])
    prices = initial * np.exp(-0.5*variance + np.sqrt(variance)*z)
    pnl = pnl_for_position(prices, initial, contracts, multiplier)
    losses = -pnl
    # Equal-weight scenarios: upper 5% average defines empirical ES.
    var = float(np.quantile(losses, confidence, method='linear'))
    tail_count = int(round((1-confidence)*scenarios))
    es = float(np.sort(losses)[-tail_count:].mean())
    # Meaningful invariants of valuation and aggregation.
    assert pnl_for_position(initial, initial, contracts, multiplier) == 0
    np.testing.assert_allclose(pnl_for_position(prices, initial, 2, multiplier), 2*pnl)
    np.testing.assert_allclose(pnl_for_position(prices, initial, -1, multiplier), -pnl)
    assert np.isfinite(pnl).all() and es >= var and len(pnl) == scenarios
    summary = {
        'purpose': 'ARCHITECTURE VALIDATION ONLY; SYNTHETIC VOLATILITY',
        'valuation_date': date, 'horizon_date': str(horizon_date.date()),
        'horizon_business_days': 1, 'horizon_calendar_days': 3,
        'year_fraction': 'ACT/365F', 'contract': position.Contract,
        'position_contracts': contracts, 'barrels_per_contract': multiplier,
        'initial_futures_price_USD_per_barrel': initial,
        'curve_converged': bool(curve.converged),
        'curve_any_bound': bool(curve.any_bound),
        'curve_RMSE_USD': float(curve.RMSE_USD),
        'kappa_X': kx, 'kappa_Y': ky,
        'sigma_X_synthetic': sigma_x, 'sigma_Y_synthetic': sigma_y,
        'rho_synthetic': rho, 'volatility_source': 'fixed illustrative assumptions; not calibrated',
        'drift_assumption': 'zero expected futures price return',
        'scenario_count': scenarios, 'seed': seed, 'confidence': confidence,
        'VaR_USD': var, 'ES_USD': es,
        'mean_PnL_USD': float(pnl.mean()),
        'futures_log_return_standard_deviation': float(np.sqrt(variance)),
        'checks': 'zero-shock PnL, quantity scaling, position sign, finite values, ES >= VaR passed',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'scenario': np.arange(scenarios), 'futures_price': prices,
                  'PnL_USD': pnl, 'loss_USD': losses}).to_csv(OUT/'scenarios.csv', index=False)
    (OUT/'risk_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report = (
        '最小风险引擎：仅用于架构验证\n'
        f'估值日 {date}；风险日 {horizon_date.date()}（下一交易日，跨周末3个日历日）。\n'
        f'头寸：做多1手 {position.Contract}，1000桶；起始价格 {initial:.2f} 美元/桶。\n'
        f'情景数量 {scenarios:,}；95% VaR = {var:,.2f} 美元；ES = {es:,.2f} 美元。\n'
        f'校准收敛：{bool(curve.converged)}；存在触界：{bool(curve.any_bound)}。\n'
        '输入波动率为固定合成假设（0.35、0.20，相关性0.30），未经真实市场校准。\n'
        '使用观察到的期货价格作为起点，校准衰减参数决定情景载荷；假设期货价格期望不变。\n'
        'ES为损失最大的5%情景的平均值；盈亏已乘头寸数量和合约乘数。\n'
        '不含交易费用、流动性、保证金及交割；尚未实现通用节假日日历或多头寸组合。\n'
        '检查通过：零价格变化盈亏为零、数量缩放、方向反转、有限值、ES不小于VaR。\n'
    )
    (OUT/'risk_report.txt').write_text(report, encoding='utf-8')
    print(f'Architecture demo complete: 95% VaR USD {var:,.2f}; ES USD {es:,.2f}.')
    print('Synthetic volatility; validation checks passed. See Output/Step4a_Minimal_Risk_Engine.')


if __name__ == '__main__':
    main()
