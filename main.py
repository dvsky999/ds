from __future__ import annotations

import asyncio
import logging

from agents.collector.binance_collector import BinanceCollector
from agents.features.feature_engine import FeatureEngine
from agents.meta.selector import MetaStrategySelector, StrategyScore
from agents.portfolio.optimizer import PortfolioOptimizer
from agents.risk.risk_engine import RiskEngine, RiskLimits
from backtests.vectorized_backtester import VectorizedBacktester
from config.settings import Settings
from database.duckdb_client import DuckDBClient
from database.repository import MarketRepository
from strategies.basic import mean_reversion_signal, trend_following_signal, volatility_breakout_signal
from utils.logging_config import configure_logging


async def run() -> None:
    settings = Settings.load()
    configure_logging()

    db = DuckDBClient(settings.duckdb_path)
    repo = MarketRepository(db=db, parquet_root=settings.parquet_root)
    collector = BinanceCollector(settings=settings, repository=repo)

    await collector.collect_snapshot()
    klines = db.dataframe("SELECT * FROM klines ORDER BY open_time")

    features = FeatureEngine().transform(klines)

    backtester = VectorizedBacktester()
    trend = backtester.run(features["close"], trend_following_signal(features))
    mean = backtester.run(features["close"], mean_reversion_signal(features))
    vola = backtester.run(features["close"], volatility_breakout_signal(features))

    scores = [
        StrategyScore("trend_following", trend.sharpe, trend.drawdown, trend.win_rate),
        StrategyScore("mean_reversion", mean.sharpe, mean.drawdown, mean.win_rate),
        StrategyScore("volatility_breakout", vola.sharpe, vola.drawdown, vola.win_rate),
    ]
    ranked = MetaStrategySelector().rank(scores)
    selected = MetaStrategySelector().choose(features.iloc[-1]["market_regime"], ranked)

    optimizer = PortfolioOptimizer()
    risk_engine = RiskEngine(RiskLimits(settings.max_daily_loss, settings.max_drawdown, 0.3))
    strategy_returns = {
        "trend_following": features["close"].pct_change().fillna(0.0) * trend_following_signal(features).shift(1).fillna(0.0),
        "mean_reversion": features["close"].pct_change().fillna(0.0) * mean_reversion_signal(features).shift(1).fillna(0.0),
        "volatility_breakout": features["close"].pct_change().fillna(0.0) * volatility_breakout_signal(features).shift(1).fillna(0.0),
    }
    weights = optimizer.risk_parity_weights(__import__("pandas").DataFrame(strategy_returns))

    logging.getLogger(__name__).info("Selected strategy: %s", selected)
    logging.getLogger(__name__).info("Dynamic weights: %s", weights.to_dict())
    logging.getLogger(__name__).info("Risk kill switch active: %s", risk_engine.kill_switch)


if __name__ == "__main__":
    asyncio.run(run())
