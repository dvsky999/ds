from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aiohttp
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import Settings
from database.repository import MarketRepository

LOGGER = logging.getLogger(__name__)


class BinanceCollector:
    def __init__(self, settings: Settings, repository: MarketRepository) -> None:
        self.settings = settings
        self.repository = repository

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _get(self, endpoint: str, params: dict[str, Any]) -> Any:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.settings.binance_rest_url}{endpoint}", params=params, timeout=20) as response:
                response.raise_for_status()
                return await response.json()

    async def fetch_klines(self, limit: int = 500) -> pd.DataFrame:
        payload = await self._get("/fapi/v1/klines", {"symbol": self.settings.symbol, "interval": self.settings.interval, "limit": limit})
        frame = pd.DataFrame(payload, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"])
        for col in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        frame = frame.dropna().drop_duplicates(subset=["open_time"]).sort_values("open_time")
        return frame

    async def fetch_funding_rates(self, limit: int = 200) -> pd.DataFrame:
        payload = await self._get("/fapi/v1/fundingRate", {"symbol": self.settings.symbol, "limit": limit})
        frame = pd.DataFrame(payload)
        if frame.empty:
            return frame
        frame["fundingRate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
        frame["fundingTime"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True)
        return frame.dropna().drop_duplicates()

    async def fetch_open_interest(self) -> pd.DataFrame:
        payload = await self._get("/fapi/v1/openInterest", {"symbol": self.settings.symbol})
        frame = pd.DataFrame([payload])
        frame["openInterest"] = pd.to_numeric(frame["openInterest"], errors="coerce")
        frame["time"] = pd.to_datetime(frame["time"], unit="ms", utc=True)
        return frame.dropna()

    async def fetch_trades(self, limit: int = 1000) -> pd.DataFrame:
        payload = await self._get("/fapi/v1/trades", {"symbol": self.settings.symbol, "limit": limit})
        frame = pd.DataFrame(payload)
        if frame.empty:
            return frame
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        frame["qty"] = pd.to_numeric(frame["qty"], errors="coerce")
        frame["time"] = pd.to_datetime(frame["time"], unit="ms", utc=True)
        return frame.dropna().drop_duplicates(subset=["id"])

    async def collect_snapshot(self) -> None:
        klines, trades, funding, oi = await asyncio.gather(
            self.fetch_klines(),
            self.fetch_trades(),
            self.fetch_funding_rates(),
            self.fetch_open_interest(),
        )
        self.repository.append_parquet("klines", klines)
        self.repository.append_parquet("trades", trades)
        if not funding.empty:
            self.repository.append_parquet("funding_rates", funding)
        self.repository.append_parquet("open_interest", oi)
        self.repository.upsert_duckdb("klines", klines)
        self.repository.upsert_duckdb("trades", trades)

    async def stream_trades(self) -> None:
        stream = f"{self.settings.symbol.lower()}@trade"
        ws_url = f"{self.settings.binance_ws_url}/{stream}"
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ws_url, heartbeat=20) as ws:
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                payload = json.loads(msg.data)
                                trade = pd.DataFrame([
                                    {
                                        "id": payload.get("t"),
                                        "price": float(payload.get("p", 0.0)),
                                        "qty": float(payload.get("q", 0.0)),
                                        "time": pd.to_datetime(payload.get("T"), unit="ms", utc=True),
                                    }
                                ])
                                self.repository.append_parquet("trades", trade)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Trade stream disconnected, retrying: %s", exc)
                await asyncio.sleep(3)
