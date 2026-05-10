from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass(slots=True)
class EnvConfig:
    fee_rate: float = 0.0004
    drawdown_penalty: float = 0.1


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, frame: pd.DataFrame, feature_columns: list[str], config: EnvConfig | None = None) -> None:
        super().__init__()
        self.frame = frame.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.config = config or EnvConfig()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(feature_columns),), dtype=np.float32)
        self.index = 0
        self.position = 0
        self.cash = 1.0
        self.equity_peak = 1.0

    def _observation(self) -> np.ndarray:
        return self.frame.loc[self.index, self.feature_columns].to_numpy(dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.index = 0
        self.position = 0
        self.cash = 1.0
        self.equity_peak = 1.0
        return self._observation(), {}

    def step(self, action: int):
        prev_price = float(self.frame.loc[self.index, "close"])
        self.index += 1
        done = self.index >= len(self.frame) - 1
        next_price = float(self.frame.loc[self.index, "close"])

        if action == 1:
            self.position = 1
        elif action == 2:
            self.position = -1
        elif action == 3:
            self.position = 0

        pnl = self.position * ((next_price - prev_price) / prev_price)
        fee = self.config.fee_rate if action in {1, 2, 3} else 0.0
        self.cash *= 1 + pnl - fee
        self.equity_peak = max(self.equity_peak, self.cash)
        drawdown = max(0.0, (self.equity_peak - self.cash) / self.equity_peak)

        reward = pnl - fee - self.config.drawdown_penalty * drawdown
        if self.index > 10:
            window = self.frame.loc[self.index - 10 : self.index, "returns"].to_numpy(dtype=float)
            sharpe = float(np.mean(window) / (np.std(window) + 1e-8))
            reward += sharpe * 0.01

        return self._observation(), float(reward), done, False, {"equity": self.cash, "drawdown": drawdown}
