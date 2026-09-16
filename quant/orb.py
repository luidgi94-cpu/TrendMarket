"""Opening Range Breakout XAUUSD, calibre pour un compte de 450 USD.

Contrainte structurante et souvent ignoree : le lot minimum est 0.01
(= 1 once). On ne peut PAS descendre en dessous. Donc si le stop naturel
d'un setup implique un risque superieur au budget, le trade doit etre
REFUSE - impossible de reduire la taille. Ce filtre elimine une part
significative des signaux et c'est la principale raison pour laquelle le
taux de trades/jour reel est bien inferieur au nombre de signaux bruts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Session:
    name: str
    range_start: str   # UTC "HH:MM"
    range_end: str
    trade_end: str


SESSIONS = (
    Session("London", "06:00", "08:00", "11:00"),
    Session("NewYork", "12:30", "13:30", "16:00"),
)


@dataclass
class ORBConfig:
    capital: float = 450.0
    lot: float = 0.01                 # 0.01 lot = 1 once -> PnL = mouvement en USD
    oz_per_lot: float = 100.0
    spread_usd: float = 0.25          # cout aller-retour par once
    slippage_usd: float = 0.10        # glissement sur ordre stop, par once
    max_risk_pct: float = 0.02        # plafond de risque par trade
    rr_target: float = 2.0
    partial_at_r: float = 1.5
    partial_fraction: float = 0.5
    atr_stop_cap: float = 1.5         # stop <= 1.5 x ATR(H1)
    atr_pct_floor: float = 30.0       # percentile de vol mini (anti-compression)
    use_daily_bias: bool = True
    max_trades_per_day: int = 2
    daily_stop_pct: float = 0.04
    weekly_stop_pct: float = 0.06
    sessions: tuple[Session, ...] = field(default_factory=lambda: SESSIONS)

    @property
    def units(self) -> float:
        """Onces reellement tradees (PnL en USD = mouvement x units)."""
        return self.lot * self.oz_per_lot


def hourly_atr(m15: pd.DataFrame, window: int = 24) -> pd.Series:
    h1 = m15.resample("1h").agg({"open": "first", "high": "max",
                                 "low": "min", "close": "last"}).dropna()
    prev = h1.close.shift(1)
    tr = pd.concat([h1.high - h1.low, (h1.high - prev).abs(),
                    (h1.low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window // 2).mean()


def daily_bias(m15: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    """Biais de tendance journalier, decale d'un jour (pas de lookahead).

    Meme logique que le systeme swing : on ne prend que les breakouts
    alignes avec la tendance du timeframe superieur. Un breakout contre la
    tendance D1 est le setup le plus cher du marche.
    """
    d1 = m15.close.resample("1D").last().dropna()
    bias = np.sign(d1.ewm(span=fast).mean() - d1.ewm(span=slow).mean())
    return bias.shift(1)


def _atr_percentile(atr: pd.Series, window: int = 480) -> pd.Series:
    return atr.rolling(window, min_periods=100).rank(pct=True) * 100.0


def backtest_orb(m15: pd.DataFrame, cfg: ORBConfig) -> tuple[pd.DataFrame, dict]:
    atr_h1 = hourly_atr(m15)
    atr_pct = _atr_percentile(atr_h1)
    bias = daily_bias(m15)
    units = cfg.units

    equity = cfg.capital
    trades: list[dict] = []
    rejects = {"biais_oppose": 0, "vol_trop_basse": 0, "risque_trop_grand": 0,
               "pas_de_cassure": 0, "stop_journalier": 0, "stop_hebdo": 0}

    week_start_equity = equity
    current_week = None

    for day, day_bars in m15.groupby(m15.index.normalize()):
        if len(day_bars) < 40:
            continue
        iso_week = day.isocalendar()[:2]
        if iso_week != current_week:
            current_week, week_start_equity = iso_week, equity
        if equity < week_start_equity * (1 - cfg.weekly_stop_pct):
            rejects["stop_hebdo"] += 1
            continue

        day_start_equity = equity
        day_bias = bias.get(day, 0.0)
        if np.isnan(day_bias):
            day_bias = 0.0
        n_today = 0

        for sess in cfg.sessions:
            if n_today >= cfg.max_trades_per_day:
                break
            if equity < day_start_equity * (1 - cfg.daily_stop_pct):
                rejects["stop_journalier"] += 1
                break

            rng_bars = day_bars.between_time(sess.range_start, sess.range_end)
            win_bars = day_bars.between_time(sess.range_end, sess.trade_end)
            if len(rng_bars) < 3 or len(win_bars) < 3:
                continue

            hi, lo = rng_bars.high.max(), rng_bars.low.min()
            ts0 = win_bars.index[0]
            atr_now = atr_h1.asof(ts0)
            pct_now = atr_pct.asof(ts0)
            if np.isnan(atr_now) or np.isnan(pct_now):
                continue
            if pct_now < cfg.atr_pct_floor:
                rejects["vol_trop_basse"] += 1
                continue

            # --- recherche de la cassure ---------------------------------
            entry = direction = None
            entry_ts = None
            for ts, bar in win_bars.iterrows():
                if bar.close > hi:
                    direction, entry, entry_ts = 1, bar.close, ts
                    break
                if bar.close < lo:
                    direction, entry, entry_ts = -1, bar.close, ts
                    break
            if direction is None:
                rejects["pas_de_cassure"] += 1
                continue

            if cfg.use_daily_bias and day_bias != 0 and direction != day_bias:
                rejects["biais_oppose"] += 1
                continue

            # --- stop : cote oppose du range, plafonne en ATR -------------
            raw_stop = lo if direction > 0 else hi
            stop_dist = abs(entry - raw_stop)
            stop_dist = min(stop_dist, cfg.atr_stop_cap * atr_now)
            stop_dist = max(stop_dist, 0.3 * atr_now)  # jamais absurdement serre

            risk_usd = stop_dist * units + (cfg.spread_usd + cfg.slippage_usd) * units
            # LE filtre qui compte : lot minimum non reductible
            if risk_usd > equity * cfg.max_risk_pct:
                rejects["risque_trop_grand"] += 1
                continue

            stop_px = entry - direction * stop_dist
            partial_px = entry + direction * cfg.partial_at_r * stop_dist
            target_px = entry + direction * cfg.rr_target * stop_dist

            # --- simulation barre par barre -------------------------------
            after = win_bars.loc[win_bars.index > entry_ts]
            if after.empty:
                continue
            remaining = 1.0
            realised = 0.0
            partial_done = False
            exit_reason, exit_px = "fin_session", after.close.iloc[-1]

            for ts, bar in after.iterrows():
                hit_stop = (bar.low <= stop_px) if direction > 0 else (bar.high >= stop_px)
                hit_tp = (bar.high >= target_px) if direction > 0 else (bar.low <= target_px)
                hit_partial = (bar.high >= partial_px) if direction > 0 else (bar.low <= partial_px)

                # hypothese conservatrice : si stop ET cible dans la meme
                # bougie, on suppose que le STOP est touche en premier
                if hit_stop:
                    realised += remaining * (stop_px - entry) * direction
                    remaining, exit_reason, exit_px = 0.0, \
                        ("stop_apres_partiel" if partial_done else "stop"), stop_px
                    break
                if hit_partial and not partial_done:
                    realised += cfg.partial_fraction * (partial_px - entry) * direction
                    remaining -= cfg.partial_fraction
                    partial_done = True
                    stop_px = entry  # on passe a breakeven
                if hit_tp:
                    realised += remaining * (target_px - entry) * direction
                    remaining, exit_reason, exit_px = 0.0, "cible", target_px
                    break

            if remaining > 0:
                realised += remaining * (exit_px - entry) * direction

            gross = realised * units
            cost = (cfg.spread_usd + cfg.slippage_usd) * units
            pnl = gross - cost
            equity += pnl
            n_today += 1

            trades.append({
                "date": day, "session": sess.name, "entry_ts": entry_ts,
                "direction": "long" if direction > 0 else "short",
                "entry": entry, "stop_dist": stop_dist,
                "risk_usd": stop_dist * units, "pnl": pnl,
                "R": pnl / (stop_dist * units) if stop_dist > 0 else 0.0,
                "exit": exit_reason, "equity": equity,
            })

            if equity < 50:
                break
        if equity < 50:
            break

    df = pd.DataFrame(trades)
    n_days = m15.index.normalize().nunique()
    stats = {
        "jours_de_marche": n_days,
        "nb_trades": len(df),
        "trades_par_jour": len(df) / n_days if n_days else 0.0,
        "capital_final": equity,
        "rendement_total": equity / cfg.capital - 1.0,
    }
    if not df.empty:
        wins = df[df.pnl > 0]
        stats |= {
            "winrate": len(wins) / len(df),
            "R_moyen": df.R.mean(),
            "R_median": df.R.median(),
            "gain_moyen_R": wins.R.mean() if len(wins) else 0.0,
            "perte_moyenne_R": df[df.pnl <= 0].R.mean() if len(df) > len(wins) else 0.0,
            "max_DD": float((df.equity / df.equity.cummax() - 1).min()),
            "cout_total": len(df) * (cfg.spread_usd + cfg.slippage_usd) * units,
        }
    stats["rejets"] = rejects
    return df, stats
