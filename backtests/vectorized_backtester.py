from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class BacktestResult:
    sharpe: float
    sortino: float
    win_rate: float
    profit_factor: float
    drawdown: float
    expectancy: float
    equity_curve: pd.Series


class VectorizedBacktester:
    def run(self, prices: pd.Series, signal: pd.Series, fee_rate: float = 0.0004) -> BacktestResult:
        returns = prices.pct_change().fillna(0.0)
        strategy_returns = signal.shift(1).fillna(0.0) * returns
        fees = signal.diff().abs().fillna(0.0) * fee_rate
        net = strategy_returns - fees
        equity = (1 + net).cumprod()
        dd = (equity.cummax() - equity) / equity.cummax().replace(0.0, np.nan)

        positive = net[net > 0]
        negative = net[net < 0]
        sharpe = float(net.mean() / (net.std() + 1e-8) * np.sqrt(365))
        sortino = float(net.mean() / (negative.std() + 1e-8) * np.sqrt(365))
        win_rate = float((net > 0).mean())
        profit_factor = float(positive.sum() / abs(negative.sum() + 1e-8))
        expectancy = float(net.mean())
        drawdown = float(dd.max())

        return BacktestResult(
            sharpe=sharpe,
            sortino=sortino,
            win_rate=win_rate,
            profit_factor=profit_factor,
            drawdown=drawdown,
            expectancy=expectancy,
            equity_curve=equity,
        )
