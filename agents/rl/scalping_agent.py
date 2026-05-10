from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.rl.environment import TradingEnv

try:
    from stable_baselines3 import PPO
except Exception:  # noqa: BLE001
    PPO = None


class ScalpingAgent:
    def __init__(self, model_dir: str = "models") -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def train(self, frame: pd.DataFrame, feature_columns: list[str], timesteps: int = 20_000) -> None:
        if PPO is None:
            raise RuntimeError("stable-baselines3 is required for training")
        env = TradingEnv(frame, feature_columns)
        self.model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="logs/tensorboard")
        self.model.learn(total_timesteps=timesteps)

    def save(self, name: str = "scalping_agent") -> str:
        if self.model is None:
            raise RuntimeError("Model not trained")
        path = self.model_dir / name
        self.model.save(str(path))
        return str(path)

    def load(self, path: str) -> None:
        if PPO is None:
            raise RuntimeError("stable-baselines3 is required for loading")
        self.model = PPO.load(path)

    def act(self, observation):
        if self.model is None:
            return 0
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)
