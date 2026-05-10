from __future__ import annotations

import pandas as pd

from agents.rl.scalping_agent import ScalpingAgent


class RLTrainingPipeline:
    def __init__(self) -> None:
        self.agent = ScalpingAgent()

    def run(self, frame: pd.DataFrame, feature_columns: list[str], timesteps: int = 20_000) -> str:
        self.agent.train(frame=frame, feature_columns=feature_columns, timesteps=timesteps)
        return self.agent.save()
