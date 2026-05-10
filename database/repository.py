from __future__ import annotations

from pathlib import Path

import pandas as pd

from database.duckdb_client import DuckDBClient


class MarketRepository:
    def __init__(self, db: DuckDBClient, parquet_root: str) -> None:
        self.db = db
        self.parquet_root = Path(parquet_root)
        self.parquet_root.mkdir(parents=True, exist_ok=True)

    def append_parquet(self, dataset: str, frame: pd.DataFrame) -> Path:
        path = self.parquet_root / f"{dataset}.parquet"
        if path.exists():
            previous = pd.read_parquet(path)
            frame = pd.concat([previous, frame], ignore_index=True)
            frame = frame.drop_duplicates()
        frame.to_parquet(path, index=False)
        return path

    def upsert_duckdb(self, table: str, frame: pd.DataFrame) -> None:
        with self.db.connect() as conn:
            conn.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM frame LIMIT 0")
            conn.register("incoming", frame)
            conn.execute(f"INSERT INTO {table} SELECT * FROM incoming")
