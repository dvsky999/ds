from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb


class DuckDBClient:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.path)

    def execute(self, query: str, params: tuple[Any, ...] | None = None) -> None:
        with self.connect() as conn:
            if params is None:
                conn.execute(query)
            else:
                conn.execute(query, params)

    def dataframe(self, query: str):
        with self.connect() as conn:
            return conn.execute(query).fetchdf()
