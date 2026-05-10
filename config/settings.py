from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    app_env: str = "dev"
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    duckdb_path: str = "database/market.duckdb"
    parquet_root: str = "data/raw"
    binance_rest_url: str = "https://fapi.binance.com"
    binance_ws_url: str = "wss://fstream.binance.com/ws"
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.15
    base_currency_usdt: float = 10000.0

    @classmethod
    def load(cls, env_file: str = ".env") -> "Settings":
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path)
        return cls(
            app_env=os.getenv("APP_ENV", "dev"),
            symbol=os.getenv("SYMBOL", "BTCUSDT"),
            interval=os.getenv("INTERVAL", "1m"),
            duckdb_path=os.getenv("DUCKDB_PATH", "database/market.duckdb"),
            parquet_root=os.getenv("PARQUET_ROOT", "data/raw"),
            binance_rest_url=os.getenv("BINANCE_REST_URL", "https://fapi.binance.com"),
            binance_ws_url=os.getenv("BINANCE_WS_URL", "wss://fstream.binance.com/ws"),
            max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "0.05")),
            max_drawdown=float(os.getenv("MAX_DRAWDOWN", "0.15")),
            base_currency_usdt=float(os.getenv("BASE_CURRENCY_USDT", "10000")),
        )
