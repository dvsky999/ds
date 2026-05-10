import unittest

import pandas as pd

from agents.risk.risk_engine import RiskEngine, RiskLimits
from backtests.vectorized_backtester import VectorizedBacktester


class RiskBacktestTests(unittest.TestCase):
    def test_risk_engine_kill_switch(self) -> None:
        engine = RiskEngine(RiskLimits(max_daily_loss=0.05, max_drawdown=0.2, max_exposure=0.5))
        ok = engine.validate(daily_pnl=-0.01, drawdown=0.1, exposure=0.1)
        self.assertTrue(ok)
        blocked = engine.validate(daily_pnl=-0.08, drawdown=0.1, exposure=0.1)
        self.assertFalse(blocked)

    def test_backtester_returns_metrics(self) -> None:
        prices = pd.Series([100, 101, 102, 101, 103, 104])
        signal = pd.Series([0, 1, 1, 0, 1, 1])
        result = VectorizedBacktester().run(prices=prices, signal=signal)
        self.assertIsInstance(result.sharpe, float)
        self.assertEqual(len(result.equity_curve), len(prices))


if __name__ == "__main__":
    unittest.main()
