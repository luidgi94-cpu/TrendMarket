"""Chargement OHLC et reechantillonnage multi-timeframe."""
import csv
from datetime import datetime, timezone, timedelta
from typing import List, Dict
from .model import Candle

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}


def _parse_ts(raw: str) -> datetime:
    raw = raw.strip().replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y%m%d %H%M%S"):
        try:
            dt = datetime.fromisoformat(raw) if fmt is None else datetime.strptime(raw, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"horodatage illisible: {raw!r}")


def _sniff(path: str) -> str:
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        head = fh.readline()
    return "\t" if head.count("\t") >= 2 else (";" if head.count(";") > head.count(",") else ",")


def load_csv(path: str) -> List[Candle]:
    """CSV ou TSV avec en-tete : time/date, open, high, low, close[, volume].

    Gere l'export MetaTrader 5 (tabulations, en-tetes <DATE> <TIME> ...,
    colonnes DATE et TIME separees) comme les exports classiques.
    Les horodatages sans fuseau sont traites comme UTC.
    """
    out: List[Candle] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh, delimiter=_sniff(path))
        # <DATE> -> date
        cols = {c.lower().strip().strip("<>"): c for c in (rd.fieldnames or [])}

        def col(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            raise KeyError(f"colonne absente, candidats {names}, trouve {list(cols)}")

        c_ts = col("time", "timestamp", "date", "datetime", "gmt time")
        # MT5 : DATE et TIME dans deux colonnes distinctes
        c_t2 = cols.get("time") if "date" in cols and "time" in cols else None
        if c_t2 and c_ts == c_t2:
            c_ts, c_t2 = cols["date"], cols["time"]
        c_o, c_h = col("open"), col("high")
        c_l, c_c = col("low"), col("close")
        c_v = cols.get("tickvol") or cols.get("volume") or cols.get("vol")
        c_sp = cols.get("spread")
        for row in rd:
            try:
                raw_ts = row[c_ts] if not c_t2 else f"{row[c_ts]} {row[c_t2]}"
                out.append(Candle(
                    ts=_parse_ts(raw_ts),
                    open=float(row[c_o]), high=float(row[c_h]),
                    low=float(row[c_l]), close=float(row[c_c]),
                    volume=float(row[c_v]) if c_v and row.get(c_v) else 0.0,
                    spread=float(row[c_sp]) if c_sp and row.get(c_sp) else 0.0,
                ))
            except (ValueError, KeyError, TypeError):
                continue  # ligne corrompue -> ignoree
    out.sort(key=lambda c: c.ts)
    return out


def resample(candles: List[Candle], tf: str) -> List[Candle]:
    """Aggrege vers un TF superieur. Les bougies sont alignees sur
    les multiples du TF depuis minuit UTC."""
    if tf not in TF_MINUTES:
        raise ValueError(f"TF inconnu: {tf}")
    step = TF_MINUTES[tf]
    buckets: Dict[datetime, List[Candle]] = {}
    for c in candles:
        mins = c.ts.hour * 60 + c.ts.minute
        anchor = c.ts.replace(hour=0, minute=0, second=0, microsecond=0) \
                 + timedelta(minutes=(mins // step) * step)
        buckets.setdefault(anchor, []).append(c)

    out: List[Candle] = []
    for anchor in sorted(buckets):
        grp = buckets[anchor]
        out.append(Candle(
            ts=anchor, open=grp[0].open,
            high=max(x.high for x in grp), low=min(x.low for x in grp),
            close=grp[-1].close, volume=sum(x.volume for x in grp),
        ))
    return out


def atr(candles: List[Candle], period: int = 14) -> List[float]:
    """ATR de Wilder. atr[i] n'utilise que les bougies <= i."""
    if not candles:
        return []
    trs = [candles[0].high - candles[0].low]
    for i in range(1, len(candles)):
        p, c = candles[i - 1], candles[i]
        trs.append(max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close)))
    out = [0.0] * len(candles)
    run = trs[0]
    for i, tr in enumerate(trs):
        run = tr if i == 0 else (run * (period - 1) + tr) / period
        out[i] = run
    return out
