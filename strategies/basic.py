from __future__ import annotations

import pandas as pd


def trend_following_signal(frame: pd.DataFrame) -> pd.Series:
    return (frame["roll_mean_5"] > frame["roll_mean_20"]).astype(int)


def mean_reversion_signal(frame: pd.DataFrame) -> pd.Series:
    return (frame["close"] < frame["roll_mean_20"]).astype(int) - (frame["close"] > frame["roll_mean_20"]).astype(int)


def volatility_breakout_signal(frame: pd.DataFrame) -> pd.Series:
    threshold = frame["high"].rolling(20, min_periods=1).max().shift(1)
    return (frame["close"] > threshold).astype(int)
