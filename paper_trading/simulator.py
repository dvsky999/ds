from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TradeRecord:
    action: int
    price: float
    pnl: float


@dataclass(slots=True)
class PaperPortfolio:
    cash: float
    records: list[TradeRecord] = field(default_factory=list)

    def log_trade(self, action: int, price: float, pnl: float) -> None:
        self.cash += pnl
        self.records.append(TradeRecord(action=action, price=price, pnl=pnl))
