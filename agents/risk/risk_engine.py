from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RiskLimits:
    max_daily_loss: float
    max_drawdown: float
    max_exposure: float


class RiskEngine:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self.kill_switch = False

    def position_size(self, equity: float, volatility: float, risk_per_trade: float = 0.01) -> float:
        if volatility <= 1e-8:
            return 0.0
        raw = equity * risk_per_trade / volatility
        return min(raw, equity * self.limits.max_exposure)

    def validate(self, daily_pnl: float, drawdown: float, exposure: float) -> bool:
        if daily_pnl <= -self.limits.max_daily_loss or drawdown >= self.limits.max_drawdown or exposure > self.limits.max_exposure:
            self.kill_switch = True
        return not self.kill_switch
