"""Phase 1: Data acquisition and preprocessing for BTC/USDT trend-following research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd


@dataclass
class DataConfig:
    """Configuration for downloading and cleaning OHLCV data."""

    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    years: int = 4
    exchange_id: str = "binance"
    limit_per_call: int = 1000


class BTCDataPipeline:
    """Downloads and preprocesses BTC/USDT OHLCV data using ccxt."""

    def __init__(self, config: Optional[DataConfig] = None) -> None:
        self.config = config or DataConfig()
        self.exchange = self._build_exchange()

    def _build_exchange(self) -> ccxt.Exchange:
        exchange_class = getattr(ccxt, self.config.exchange_id)
        return exchange_class({"enableRateLimit": True})

    def _start_timestamp_ms(self) -> int:
        now = datetime.now(timezone.utc)
        start = now - pd.DateOffset(years=self.config.years)
        start_dt = pd.Timestamp(start).to_pydatetime()
        return int(start_dt.timestamp() * 1000)

    def fetch_ohlcv(self) -> pd.DataFrame:
        since = self._start_timestamp_ms()
        all_rows = []

        while True:
            batch = self.exchange.fetch_ohlcv(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                since=since,
                limit=self.config.limit_per_call,
            )
            if not batch:
                break

            all_rows.extend(batch)
            last_ts = batch[-1][0]
            next_since = last_ts + 1

            if next_since <= since:
                break
            since = next_since

            if len(batch) < self.config.limit_per_call:
                break

        if not all_rows:
            raise ValueError("No OHLCV data returned from exchange.")

        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(all_rows, columns=columns)
        return df

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        clean = df.copy()
        clean["timestamp"] = pd.to_datetime(clean["timestamp"], unit="ms", utc=True)
        clean = clean.drop_duplicates(subset=["timestamp"], keep="last")
        clean = clean.sort_values("timestamp")
        clean = clean.set_index("timestamp")

        clean = clean[~clean.index.duplicated(keep="last")]
        clean = clean[clean.index.notna()]
        clean = clean.dropna(how="any")

        if not clean.index.is_monotonic_increasing:
            clean = clean.sort_index()

        return clean

    def run(self, save_path: Optional[str] = None) -> pd.DataFrame:
        raw = self.fetch_ohlcv()
        processed = self.preprocess(raw)

        if save_path:
            processed.to_csv(save_path)

        return processed


if __name__ == "__main__":
    pipeline = BTCDataPipeline(DataConfig(years=4, timeframe="1h"))
    data = pipeline.run(save_path="data/btcusdt_1h.csv")
    print(data.tail())
    print(f"Rows: {len(data)}")
