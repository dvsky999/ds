from __future__ import annotations

from dataclasses import dataclass

from agents.risk.risk_engine import RiskEngine


@dataclass(slots=True)
class Position:
    side: int = 0
    quantity: float = 0.0
    entry_price: float = 0.0


class PaperExecutor:
    def __init__(self, risk_engine: RiskEngine, fee_rate: float = 0.0004) -> None:
        self.risk_engine = risk_engine
        self.fee_rate = fee_rate
        self.position = Position()

    def execute(self, action: int, price: float, equity: float, volatility: float, pnl_day: float, drawdown: float) -> float:
        exposure = abs(self.position.quantity * price) / max(equity, 1e-8)
        if not self.risk_engine.validate(pnl_day, drawdown, exposure):
            self.position = Position()
            return 0.0

        if action == 0:
            return 0.0
        if action == 3:
            realized = (price - self.position.entry_price) * self.position.quantity * self.position.side
            self.position = Position()
            return realized - abs(realized) * self.fee_rate

        side = 1 if action == 1 else -1
        qty = self.risk_engine.position_size(equity, volatility) / max(price, 1e-8)
        self.position = Position(side=side, quantity=qty, entry_price=price)
        return -qty * price * self.fee_rate
