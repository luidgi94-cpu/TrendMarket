"""Chargement des donnees : Yahoo Finance, Binance, ou CSV local.

Les trois sources sont interchangeables. En environnement sans acces reseau,
le loader bascule automatiquement sur le cache local puis sur les CSV.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config import BY_KEY, MACRO_TICKERS

CACHE = Path(os.environ.get("QUANT_CACHE", Path(__file__).parent.parent / "data_cache"))
CACHE.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str) -> Path:
    return CACHE / f"{name}.csv"


def _read_cache(name: str) -> pd.Series | None:
    p = _cache_path(name)
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df.iloc[:, 0].astype(float)


def _write_cache(name: str, series: pd.Series) -> None:
    series.to_frame("value").to_csv(_cache_path(name))


def fetch_yahoo(ticker: str, start: str = "2010-01-01") -> pd.Series | None:
    """Clotures ajustees quotidiennes. Retourne None si indisponible."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.astype(float).dropna()
    except Exception:
        return None


def fetch_binance_funding(symbol: str, limit_pages: int = 40) -> pd.Series | None:
    """Historique du funding rate perpetuel, annualise.

    Binance verse le funding toutes les 8h -> 1095 periodes par an.
    """
    try:
        import requests
    except ImportError:
        return None
    rows, end_time = [], None
    try:
        for _ in range(limit_pages):
            params = {"symbol": symbol, "limit": 1000}
            if end_time:
                params["endTime"] = end_time
            r = requests.get("https://fapi.binance.com/fapi/v1/fundingRate",
                             params=params, timeout=20)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows = batch + rows
            end_time = batch[0]["fundingTime"] - 1
            if len(batch) < 1000:
                break
    except Exception:
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df.fundingTime, unit="ms")
    s = df.set_index("ts").fundingRate.astype(float) * 1095.0  # annualise
    return s.resample("D").mean().dropna()


def load_prices(keys: list[str], start: str = "2010-01-01",
                use_cache: bool = True, refresh: bool = False) -> pd.DataFrame:
    """Matrice de prix. Priorite : cache -> reseau -> CSV manuel."""
    series = {}
    for key in keys:
        inst = BY_KEY[key]
        s = None if refresh else (_read_cache(key) if use_cache else None)
        if s is None:
            s = fetch_yahoo(inst.yahoo, start)
            if s is not None and use_cache:
                _write_cache(key, s)
        if s is None:
            s = _read_cache(key)
        if s is None:
            print(f"  [!] {key} ({inst.yahoo}) indisponible - instrument ignore")
            continue
        series[key] = s
    if not series:
        raise RuntimeError(
            "Aucune donnee de prix. Lance avec acces reseau, ou depose des CSV "
            f"(colonnes: date,value) dans {CACHE}"
        )
    df = pd.DataFrame(series).sort_index()
    # ffill borne : comble les jours feries futures sans inventer de longues
    # periodes de prix fige (crypto cote 7j/7, les futures non)
    return df.ffill(limit=5).dropna(how="all")


def load_funding(keys: list[str], refresh: bool = False) -> pd.DataFrame | None:
    series = {}
    for key in keys:
        inst = BY_KEY[key]
        if inst.binance is None:
            continue
        name = f"funding_{key}"
        s = None if refresh else _read_cache(name)
        if s is None:
            s = fetch_binance_funding(inst.binance)
            if s is not None:
                _write_cache(name, s)
        if s is not None:
            series[key] = s
    return pd.DataFrame(series).sort_index() if series else None


def load_macro(refresh: bool = False) -> pd.DataFrame | None:
    series = {}
    for name, ticker in MACRO_TICKERS.items():
        s = None if refresh else _read_cache(name)
        if s is None:
            s = fetch_yahoo(ticker)
            if s is not None:
                _write_cache(name, s)
        if s is not None:
            series[name] = s
    return pd.DataFrame(series).sort_index() if series else None
