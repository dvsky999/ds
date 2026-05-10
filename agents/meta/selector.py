from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StrategyScore:
    name: str
    sharpe: float
    drawdown: float
    win_rate: float
    enabled: bool = True


class MetaStrategySelector:
    def rank(self, scores: list[StrategyScore]) -> list[StrategyScore]:
        ranked = sorted(scores, key=lambda x: (x.sharpe, x.win_rate, -x.drawdown), reverse=True)
        for score in ranked:
            score.enabled = score.sharpe > 0.2 and score.drawdown < 0.3
        return ranked

    def choose(self, regime: str, ranked: list[StrategyScore]) -> str:
        regime_map = {
            "trending": ["trend_following", "volatility_breakout"],
            "sideways": ["mean_reversion", "scalping"],
            "high_vol": ["volatility_breakout", "scalping"],
        }
        preferred = regime_map.get(regime, [s.name for s in ranked])
        for candidate in preferred:
            for score in ranked:
                if score.name == candidate and score.enabled:
                    return candidate
        return ranked[0].name
