from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngine:
    def __init__(self, timeframe_windows: tuple[int, ...] = (5, 20, 60)) -> None:
        self.timeframe_windows = timeframe_windows

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy().sort_values("open_time").reset_index(drop=True)
        data["returns"] = data["close"].pct_change()
        data["log_returns"] = np.log(data["close"]).diff()
        data["vwap"] = (data["close"] * data["volume"]).cumsum() / data["volume"].replace(0.0, np.nan).cumsum()
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - data["close"].shift(1)).abs(),
            (data["low"] - data["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        data["atr"] = tr.rolling(14, min_periods=1).mean()
        data["volatility"] = data["returns"].rolling(20, min_periods=1).std()
        data["realized_volatility"] = np.sqrt((data["returns"] ** 2).rolling(20, min_periods=1).sum())
        data["momentum"] = data["close"].pct_change(10)
        data["volume_imbalance"] = (data["taker_buy_volume"] - (data["volume"] - data["taker_buy_volume"])) / data["volume"].replace(0.0, np.nan)
        data["order_flow_imbalance"] = data["taker_buy_quote_volume"] / data["quote_asset_volume"].replace(0.0, np.nan) - 0.5
        data["trend_strength"] = (data["close"] - data["close"].rolling(20, min_periods=1).mean()) / data["close"].rolling(20, min_periods=1).std().replace(0.0, np.nan)
        data["liquidity_zone"] = pd.cut(data["volume"], bins=3, labels=["low", "mid", "high"]).astype("string")

        vol_q = data["volatility"].quantile([0.33, 0.66]).to_list()
        data["volatility_regime"] = np.select(
            [data["volatility"] <= vol_q[0], data["volatility"].between(vol_q[0], vol_q[1], inclusive="right")],
            ["low", "mid"],
            default="high",
        )
        data["market_regime"] = np.where(data["trend_strength"].abs() > 1.0, "trending", "sideways")

        for window in self.timeframe_windows:
            data[f"roll_mean_{window}"] = data["close"].rolling(window, min_periods=1).mean()
            data[f"roll_std_{window}"] = data["close"].rolling(window, min_periods=1).std()
            data[f"roll_min_{window}"] = data["low"].rolling(window, min_periods=1).min()
            data[f"roll_max_{window}"] = data["high"].rolling(window, min_periods=1).max()

        return data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
