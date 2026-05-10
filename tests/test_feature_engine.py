import unittest

import pandas as pd

from agents.features.feature_engine import FeatureEngine


class FeatureEngineTests(unittest.TestCase):
    def test_transform_adds_required_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": pd.date_range("2024-01-01", periods=80, freq="min", tz="UTC"),
                "open": [100 + i for i in range(80)],
                "high": [101 + i for i in range(80)],
                "low": [99 + i for i in range(80)],
                "close": [100 + i for i in range(80)],
                "volume": [10 + i for i in range(80)],
                "quote_asset_volume": [1000 + i for i in range(80)],
                "taker_buy_volume": [5 + i / 2 for i in range(80)],
                "taker_buy_quote_volume": [600 + i for i in range(80)],
            }
        )
        out = FeatureEngine().transform(frame)
        self.assertIn("returns", out.columns)
        self.assertIn("market_regime", out.columns)
        self.assertIn("volatility_regime", out.columns)
        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
