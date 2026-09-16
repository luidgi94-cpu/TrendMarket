"""Modele ICT / SMC rendu DETERMINISTE et backtestable.

Le reproche habituel fait au SMC n'est pas que ce soit faux, c'est que ce
soit non specifie : "order block", "displacement", "shock" ne sont pas des
definitions, ce sont des descriptions. Deux analystes lisant le meme
graphique produisent deux trades differents, donc rien n'est mesurable.

Ce module supprime toute discretion. Chaque concept recoit une regle
mecanique unique, appliquee barre par barre, sans connaissance du futur.
On peut alors poser la seule question qui compte : est-ce que ca gagne ?

Modele implemente : "ICT 2022" - le plus documente et le plus enseigne.
  1. RAID       : balayage d'un pool de liquidite (high/low asiatique ou
                  de la veille), avec retour du prix a l'interieur.
  2. MSS        : cassure de structure dans le sens oppose au raid,
                  accompagnee d'un displacement (bougie > k x ATR).
  3. FVG        : le displacement laisse une inefficience sur 3 bougies.
  4. ENTREE     : ordre limite au milieu du FVG ("consequent encroachment").
  5. STOP       : au-dela de l'extreme du raid.
  6. CIBLE      : le pool de liquidite oppose le plus proche.
  7. KILLZONE   : le raid doit se produire en session Londres ou New York.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SMCConfig:
    capital: float = 450.0
    units: float = 1.0                # onces tradees (0.01 lot XAUUSD)
    spread_usd: float = 0.25
    slippage_usd: float = 0.10
    max_risk_pct: float = 0.02

    fractal_k: int = 2                # swing = extreme sur 2k+1 bougies
    sweep_return_bars: int = 4        # retour a l'interieur sous N bougies
    displacement_atr: float = 1.8     # corps de bougie >= k x ATR(M15)
    mss_lookback: int = 24            # fenetre de recherche du MSS apres raid
    fvg_valid_bars: int = 16          # duree de validite de l'ordre limite
    min_rr: float = 1.5               # sous ce RR, pas de trade
    max_rr: float = 5.0
    stop_buffer_atr: float = 0.2
    max_trades_per_day: int = 2
    use_killzone: bool = True
    killzones: tuple[tuple[int, int], ...] = ((7, 10), (12, 15))  # heures UTC
    use_daily_bias: bool = False      # ICT "pur" ne l'impose pas


def _fractals(h: np.ndarray, l: np.ndarray, k: int):
    """Swing highs / lows par fractale. Confirmes avec k barres de retard."""
    n = len(h)
    sh = np.zeros(n, bool)
    sl = np.zeros(n, bool)
    for i in range(k, n - k):
        w_h, w_l = h[i - k:i + k + 1], l[i - k:i + k + 1]
        if h[i] == w_h.max() and (w_h == h[i]).sum() == 1:
            sh[i] = True
        if l[i] == w_l.min() and (w_l == l[i]).sum() == 1:
            sl[i] = True
    return sh, sl


def _atr_m15(m15: pd.DataFrame, window: int = 96) -> np.ndarray:
    prev = m15.close.shift(1)
    tr = pd.concat([m15.high - m15.low, (m15.high - prev).abs(),
                    (m15.low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window // 2).mean().to_numpy()


def backtest_smc(m15: pd.DataFrame, cfg: SMCConfig,
                 atr_h1: pd.Series) -> tuple[pd.DataFrame, dict]:
    idx = m15.index
    o = m15.open.to_numpy(); h = m15.high.to_numpy()
    l = m15.low.to_numpy(); c = m15.close.to_numpy()
    hours = idx.hour.to_numpy()
    days = idx.normalize()
    atr = atr_h1.reindex(idx, method="ffill").to_numpy()
    # le displacement se mesure sur l'unite de temps du SIGNAL (M15),
    # pas sur celle du contexte (H1) : sinon le critere est inatteignable
    atr15 = _atr_m15(m15)

    sh, sl = _fractals(h, l, cfg.fractal_k)

    # --- pools de liquidite : extremes asiatiques et de la veille ---------
    day_keys = pd.Index(days)
    asia = pd.DataFrame({"h": h, "l": l}, index=idx)
    asia_mask = (hours >= 0) & (hours < 6)
    asia_hi = asia[asia_mask].groupby(day_keys[asia_mask]).h.max()
    asia_lo = asia[asia_mask].groupby(day_keys[asia_mask]).l.min()
    prev_hi = asia.groupby(day_keys).h.max().shift(1)
    prev_lo = asia.groupby(day_keys).l.min().shift(1)

    def pools(day):
        """Niveaux connus AVANT l'ouverture de session. Pas de lookahead."""
        up, dn = [], []
        for src in (asia_hi, prev_hi):
            v = src.get(day, np.nan)
            if not np.isnan(v):
                up.append(v)
        for src in (asia_lo, prev_lo):
            v = src.get(day, np.nan)
            if not np.isnan(v):
                dn.append(v)
        return up, dn

    in_kz = np.zeros(len(idx), bool)
    for a, b in cfg.killzones:
        in_kz |= (hours >= a) & (hours < b)
    if not cfg.use_killzone:
        in_kz[:] = True

    equity = cfg.capital
    trades = []
    rej = {"pas_de_MSS": 0, "pas_de_FVG": 0, "RR_insuffisant": 0,
           "risque_trop_grand": 0, "limite_non_touchee": 0, "quota_jour": 0}
    per_day: dict = {}
    i = cfg.fractal_k + 1
    n = len(idx)

    while i < n - cfg.mss_lookback - cfg.fvg_valid_bars - 2:
        day = days[i]
        if np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(atr15[i]):
            i += 1; continue
        if not in_kz[i]:
            i += 1; continue
        if per_day.get(day, 0) >= cfg.max_trades_per_day:
            i += 1; continue

        up, dn = pools(day)
        a = atr[i]

        # ---------- 1. RAID ------------------------------------------------
        raid_dir = 0
        level = np.nan
        for lv in dn:                                   # balayage des lows
            if l[i] < lv <= c[i]:
                raid_dir, level = 1, lv; break          # -> biais haussier
        if raid_dir == 0:
            for lv in up:                               # balayage des highs
                if h[i] > lv >= c[i]:
                    raid_dir, level = -1, lv; break     # -> biais baissier
        if raid_dir == 0:
            i += 1; continue

        raid_extreme = l[i] if raid_dir > 0 else h[i]

        # ---------- 2. MSS + displacement ----------------------------------
        # dernier swing oppose AVANT le raid : c'est lui qu'il faut casser
        ref = np.nan
        for j in range(i - 1, max(i - 60, cfg.fractal_k), -1):
            if raid_dir > 0 and sh[j]:
                ref = h[j]; break
            if raid_dir < 0 and sl[j]:
                ref = l[j]; break
        if np.isnan(ref):
            rej["pas_de_MSS"] += 1; i += 1; continue

        mss_at = -1
        for j in range(i + 1, min(i + cfg.mss_lookback, n - 2)):
            body = abs(c[j] - o[j])
            broke = (c[j] > ref) if raid_dir > 0 else (c[j] < ref)
            if broke and body >= cfg.displacement_atr * atr15[j]:
                mss_at = j; break
            # le raid est invalide si le prix repart chercher l'extreme
            if raid_dir > 0 and l[j] < raid_extreme:
                break
            if raid_dir < 0 and h[j] > raid_extreme:
                break
        if mss_at < 0:
            rej["pas_de_MSS"] += 1; i += 1; continue

        # ---------- 3. FVG laisse par le displacement ----------------------
        fvg = None
        for j in range(max(mss_at - 2, 1), min(mss_at + 2, n - 1)):
            if raid_dir > 0 and l[j + 1] > h[j - 1]:
                fvg = (h[j - 1], l[j + 1], j + 1); break
            if raid_dir < 0 and h[j + 1] < l[j - 1]:
                fvg = (l[j + 1], h[j - 1], j + 1); break
        if fvg is None:
            rej["pas_de_FVG"] += 1; i = mss_at + 1; continue

        lo_g, hi_g, gap_end = (min(fvg[0], fvg[1]), max(fvg[0], fvg[1]), fvg[2])
        entry_px = (lo_g + hi_g) / 2.0            # consequent encroachment

        # ---------- 5/6. stop et cible -------------------------------------
        stop_px = raid_extreme - raid_dir * cfg.stop_buffer_atr * a
        risk = abs(entry_px - stop_px)
        if risk <= 0:
            i = mss_at + 1; continue

        cands = [p for p in (up if raid_dir > 0 else dn)
                 if (p > entry_px if raid_dir > 0 else p < entry_px)]
        if not cands:
            rej["RR_insuffisant"] += 1; i = mss_at + 1; continue
        target_px = min(cands) if raid_dir > 0 else max(cands)
        rr = abs(target_px - entry_px) / risk
        if rr < cfg.min_rr:
            rej["RR_insuffisant"] += 1; i = mss_at + 1; continue
        rr = min(rr, cfg.max_rr)
        target_px = entry_px + raid_dir * rr * risk

        risk_usd = risk * cfg.units + (cfg.spread_usd + cfg.slippage_usd) * cfg.units
        if risk_usd > equity * cfg.max_risk_pct:
            rej["risque_trop_grand"] += 1; i = mss_at + 1; continue

        # ---------- 4. l'ordre limite est-il touche ? ----------------------
        fill_at = -1
        for j in range(gap_end + 1, min(gap_end + 1 + cfg.fvg_valid_bars, n)):
            touched = (l[j] <= entry_px) if raid_dir > 0 else (h[j] >= entry_px)
            if touched:
                fill_at = j; break
            broke_stop = (l[j] <= stop_px) if raid_dir > 0 else (h[j] >= stop_px)
            if broke_stop:
                break
        if fill_at < 0:
            rej["limite_non_touchee"] += 1; i = mss_at + 1; continue

        # ---------- simulation ---------------------------------------------
        exit_reason, r_mult = "fin_fenetre", 0.0
        for j in range(fill_at, min(fill_at + 96, n)):
            hit_stop = (l[j] <= stop_px) if raid_dir > 0 else (h[j] >= stop_px)
            hit_tp = (h[j] >= target_px) if raid_dir > 0 else (l[j] <= target_px)
            # hypothese conservatrice : stop prioritaire dans la meme bougie
            if hit_stop:
                exit_reason, r_mult = "stop", -1.0; break
            if hit_tp:
                exit_reason, r_mult = "cible", rr; break
        else:
            r_mult = (c[min(fill_at + 95, n - 1)] - entry_px) * raid_dir / risk

        gross = r_mult * risk * cfg.units
        cost = (cfg.spread_usd + cfg.slippage_usd) * cfg.units
        pnl = gross - cost
        equity += pnl
        per_day[day] = per_day.get(day, 0) + 1

        trades.append({
            "date": day, "entry_ts": idx[fill_at],
            "direction": "long" if raid_dir > 0 else "short",
            "entry": entry_px, "stop_dist": risk, "rr_cible": rr,
            "R": pnl / (risk * cfg.units), "pnl": pnl,
            "exit": exit_reason, "equity": equity,
        })
        if equity < 50:
            break
        i = fill_at + 1

    df = pd.DataFrame(trades)
    n_days = idx.normalize().nunique()
    stats = {"jours": n_days, "nb_trades": len(df),
             "trades_par_jour": len(df) / n_days if n_days else 0.0,
             "capital_final": equity, "rejets": rej}
    if not df.empty:
        w = df[df.pnl > 0]
        stats |= {
            "winrate": len(w) / len(df),
            "R_moyen": df.R.mean(),
            "RR_cible_median": df.rr_cible.median(),
            "stop_median_usd": df.stop_dist.median(),
            "max_DD": float((df.equity / df.equity.cummax() - 1).min()),
            "part_stops": (df.exit == "stop").mean(),
            "part_cibles": (df.exit == "cible").mean(),
        }
    return df, stats
