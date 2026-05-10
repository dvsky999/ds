from __future__ import annotations

import numpy as np
import pandas as pd


class PortfolioOptimizer:
    def risk_parity_weights(self, returns: pd.DataFrame) -> pd.Series:
        vol = returns.std().replace(0.0, np.nan)
        inv_vol = 1 / vol
        weights = inv_vol / inv_vol.sum()
        return weights.fillna(0.0)

    def sharpe_weights(self, returns: pd.DataFrame) -> pd.Series:
        sharpe = returns.mean() / returns.std().replace(0.0, np.nan)
        sharpe = sharpe.clip(lower=0.0)
        if sharpe.sum() == 0:
            return pd.Series(np.repeat(1 / len(sharpe), len(sharpe)), index=sharpe.index)
        return sharpe / sharpe.sum()

    def volatility_target_scale(self, portfolio_returns: pd.Series, target_vol: float = 0.2) -> float:
        current_vol = portfolio_returns.std() * np.sqrt(365)
        if current_vol <= 1e-8:
            return 1.0
        return float(min(2.0, max(0.0, target_vol / current_vol)))
