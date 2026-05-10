# Modular AI Trading Research System (BTCUSDT)

Production-oriented quantitative AI trading research stack for Binance BTCUSDT market data.

## What is included

- Modular architecture under `agents/` for collection, features, RL, evolution, risk, execution, portfolio and meta strategy layers.
- Binance REST + WebSocket collector with async retries, validation and incremental Parquet/DuckDB storage.
- Feature engine with vectorized quantitative features and regime labeling.
- Gymnasium trading environment and PPO-based first RL agent (ScalpingAgent).
- Vectorized backtesting engine with institutional metrics.
- Risk engine, portfolio optimization, paper execution, and self-improving evolution loop.
- Dockerized runtime and environment-driven config.

## Project tree

```text
project/
├── config/
├── data/
├── database/
├── logs/
├── notebooks/
├── agents/
│   ├── collector/
│   ├── features/
│   ├── research/
│   ├── rl/
│   ├── evolution/
│   ├── risk/
│   ├── execution/
│   ├── portfolio/
│   └── meta/
├── models/
├── strategies/
├── backtests/
├── paper_trading/
├── training/
├── utils/
├── tests/
├── dashboards/
├── main.py
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

## Quick start

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Notes

- The system intentionally excludes sentiment, social media and news analysis.
- Designed for VPS targets (8GB RAM, 4 cores) using async IO, vectorized operations and lightweight storage.
