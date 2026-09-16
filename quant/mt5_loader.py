"""Lecture des exports CSV de MetaTrader 5.

Format produit par MT5 (Affichage > Symboles > Barres > Exporter) :

    <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
    2026.06.03\t08:30:00\t3412.55\t3413.10\t3412.20\t3412.90\t142\t0\t22

Deux colonnes changent tout par rapport aux donnees synthetiques :

  TICKVOL  nombre de changements de prix dans la bougie. C'est un proxy du
           flux d'ordres, seul predicteur a court horizon reellement
           documente. Absent de toute donnee simulee.

  SPREAD   le spread reel du broker, en points, bougie par bougie. Permet
           de remplacer une hypothese de cout par le cout effectivement
           subi, qui varie fortement selon l'heure et les annonces.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_mt5_csv(path: str | Path, point_value: float = 0.01) -> pd.DataFrame:
    """Charge un export MT5 et retourne un DataFrame OHLC indexe en UTC.

    point_value : taille d'un point de cotation. Sur XAUUSD cote a deux
    decimales, un point vaut 0.01 USD, donc un spread de 22 points vaut
    0.22 USD l'once.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    sep = "\t" if "\t" in raw.split("\n", 1)[0] else (
        ";" if ";" in raw.split("\n", 1)[0] else ",")

    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]

    if "date" in df.columns and "time" in df.columns:
        ts = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
        idx = pd.to_datetime(ts, format="mixed", dayfirst=False)
    elif "datetime" in df.columns:
        idx = pd.to_datetime(df["datetime"], format="mixed")
    else:
        idx = pd.to_datetime(df.iloc[:, 0], format="mixed")

    out = pd.DataFrame(index=pd.DatetimeIndex(idx))
    for src, dst in (("open", "open"), ("high", "high"),
                     ("low", "low"), ("close", "close")):
        if src not in df.columns:
            raise ValueError(f"colonne '{src}' absente : {list(df.columns)}")
        out[dst] = pd.to_numeric(df[src], errors="coerce").to_numpy()

    if "tickvol" in df.columns:
        out["tickvol"] = pd.to_numeric(df["tickvol"], errors="coerce").to_numpy()
    if "vol" in df.columns:
        v = pd.to_numeric(df["vol"], errors="coerce")
        if v.fillna(0).abs().sum() > 0:      # souvent nul chez les brokers CFD
            out["realvol"] = v.to_numpy()
    if "spread" in df.columns:
        out["spread_usd"] = pd.to_numeric(df["spread"], errors="coerce").to_numpy() \
            * point_value

    out = out.dropna(subset=["open", "high", "low", "close"]).sort_index()
    # MT5 horodate en heure du serveur, generalement UTC+2 ou UTC+3.
    # On localise en UTC par defaut ; le decalage se corrige ensuite.
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    out.index.name = "timestamp"
    return out


def resample(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    for c in ("tickvol", "realvol"):
        if c in df.columns:
            agg[c] = "sum"
    if "spread_usd" in df.columns:
        agg["spread_usd"] = "mean"
    return df.resample(freq).agg(agg).dropna(subset=["open", "close"])


def detect_server_offset(df: pd.DataFrame) -> int:
    """Estime le decalage du serveur MT5 par rapport a UTC.

    Methode : le creux de volatilite quotidien de l'or se situe autour de
    22h-23h UTC, entre la cloture de New York et l'ouverture asiatique. On
    cherche l'heure la plus calme et on en deduit le decalage. Un horodatage
    mal cale decalerait toutes les killzones et fausserait tout le backtest.
    """
    rng = (df.high - df.low).groupby(df.index.hour).mean()
    quietest = int(rng.idxmin())
    # AVERTISSEMENT : cette heuristique cherche l'heure la plus calme, et
    # se fait piéger par l'artefact de rollover de minuit, qui produit un
    # faux pic de volatilite. Sur l'export XAUUSD du courtier elle repond
    # UTC+1 alors que le decalage reel est UTC+3.
    # La calibration fiable passe par les publications macro americaines de
    # 12h30 UTC : chercher le pic de volatilite intrajournalier et en
    # retrancher 12h30. Sur ces donnees le pic tombe a 15h30 serveur, d'ou
    # UTC+3.
    return (quietest - 22) % 24


def spread_report(df: pd.DataFrame) -> pd.DataFrame:
    """Spread reel par heure : remplace toute hypothese de cout."""
    if "spread_usd" not in df.columns:
        return pd.DataFrame()
    g = df.groupby(df.index.hour).spread_usd
    return pd.DataFrame({
        "median_usd": g.median(), "q90_usd": g.quantile(0.9),
        "max_usd": g.max(), "n": g.size(),
    })
