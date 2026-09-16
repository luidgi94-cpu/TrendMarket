"""Order Block scalping : haute frequence, petit TP, fort winrate vise.

Cahier des charges :
  - 3 a 4 trades par jour minimum
  - petits take-profit assumes (RR faible)
  - Order Block comme point d'interet, zone OTE NON obligatoire
  - imbalance optionnelle
  - UNE confirmation suffit parmi : meche de rejet, bougie de rejet,
    bougie englobante, micro-CHoCH
  - trades courts

Discipline anti-lookahead conservee de bout en bout :
  entree a la CLOTURE de la bougie de confirmation, simulation a partir de
  la bougie suivante, stop seul teste sur la bougie d'entree, fractales
  lues avec leur retard de confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ScalpConfig:
    capital: float = 450.0
    units: float = 1.0
    spread_usd: float = 0.25
    slippage_usd: float = 0.10

    htf_fast: int = 20
    htf_slow: int = 60
    use_trend_filter: bool = True
    fractal_k: int = 2

    # Unite de temps de detection de l'Order Block. En pratique reelle, un
    # trader SMC prend ses Order Blocks sur une unite SUPERIEURE (H1, H4,
    # voire journalier) et execute en M5. Un Order Block detecte en M5 est
    # du bruit ; un Order Block H4 est un niveau ou de vrais volumes se sont
    # traites. C'est l'ecart le plus plausible entre une implementation
    # mecanique et la pratique des traders discretionnaires.
    ob_timeframe: str = "5min"        # "5min" | "15min" | "1h" | "4h"
    impulse_bars: int = 5
    impulse_atr: float = 0.8          # seuil bas = plus de setups
    require_imbalance: bool = False

    poi_valid_bars: int = 48          # duree de vie de l'OB (4h en M5)
    confirm_bars: int = 6             # fenetre de confirmation apres touche

    # --- confirmations : UNE SEULE suffit -------------------------------
    conf_rejection_wick: bool = True  # meche >= k x corps, du bon cote
    conf_engulfing: bool = True       # bougie englobante
    conf_choch: bool = True           # cloture au-dela du micro-swing oppose
    wick_ratio: float = 1.5

    # --- leviers d'amelioration a tester -------------------------------
    min_confirmations: int = 1        # exiger 2 confirmations simultanees ?
    atr_pct_floor: float = 0.0        # percentile de vol minimum (0 = desactive)
    atr_pct_cap: float = 100.0        # percentile de vol maximum
    breakeven_at_r: float = 0.0       # >0 : stop a l'entree apres ce multiple

    # --- gestion en deux temps : partiel puis laisser courir -------------
    partial_usd: float = 0.0          # TP1 en USD/once (50 pips or = 5.00)
    partial_r: float = 0.0            # ou en multiple de R (prioritaire si >0)
    partial_fraction: float = 0.5     # part fermee au TP1
    be_after_partial: bool = True     # remonter le stop a l'equilibre ensuite
    allowed_confirmations: tuple[str, ...] = ()   # () = toutes
    allowed_hours: tuple[int, ...] = ()           # () = toutes
    direction_filter: str = "both"    # "both" | "long" | "short"
    min_trend_force: float = 0.0      # amplitude minimale de l'ecart d'EMA H1
    require_h4: bool = False          # exiger l'alignement du H4
    # Volume : proxy du flux d'ordres, seul predicteur a court horizon
    # reellement documente (Cont, Kukanov & Stoikov). Indisponible sur
    # donnees simulees, present dans un export MetaTrader 5.
    min_vol_ratio: float = 0.0        # tickvol de la bougie / mediane glissante
    min_impulse_vol: float = 0.0      # idem sur la bougie d'impulsion

    # Le stop appartient au niveau d'INVALIDATION de la structure, soit le
    # bas de l'Order Block. Si le prix y repasse, le setup est mort. Le
    # placer sous la bougie de confirmation le rend deux fois plus serre :
    # le cout de transaction double en proportion du risque, et le bruit
    # interne a l'Order Block suffit a sortir la position.
    stop_mode: str = "ob"             # "ob" (invalidation) ou "confirmation"
    stop_buffer_atr: float = 0.15
    tp_r: float = 1.0                 # petit TP assume
    max_hold_bars: int = 48           # trade court : 4h maximum
    max_trades_per_day: int = 6
    killzones: tuple[tuple[int, int], ...] = ((7, 16),)
    max_risk_pct: float = 0.99


def _atr(df: pd.DataFrame, w: int) -> np.ndarray:
    prev = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(),
                    (df.low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(w, min_periods=w // 2).mean().to_numpy()


def _swings(h, l, k):
    n = len(h); sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(k, n - k):
        if h[i] == h[i-k:i+k+1].max() and (h[i-k:i+k+1] == h[i]).sum() == 1: sh[i] = True
        if l[i] == l[i-k:i+k+1].min() and (l[i-k:i+k+1] == l[i]).sum() == 1: sl[i] = True
    return sh, sl


def detect_htf_obs(m5: pd.DataFrame, freq: str, impulse_atr: float,
                   impulse_bars: int, require_imbalance: bool) -> list[dict]:
    """Detecte les Order Blocks sur une unite de temps superieure.

    Retourne, pour chaque Order Block : sa direction, ses bornes, et
    l'horodatage a partir duquel il devient exploitable en M5 (la cloture
    de la bougie d'impulsion qui l'a valide, jamais avant).
    """
    htf = m5.resample(freq).agg({"open": "first", "high": "max",
                                 "low": "min", "close": "last"}).dropna()
    if len(htf) < 50:
        return []
    o = htf.open.to_numpy(); h = htf.high.to_numpy()
    l = htf.low.to_numpy(); c = htf.close.to_numpy()
    prev = htf.close.shift(1)
    tr = pd.concat([htf.high - htf.low, (htf.high - prev).abs(),
                    (htf.low - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(48, min_periods=20).mean().to_numpy()

    obs = []
    for i in range(impulse_bars + 2, len(htf)):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        move = c[i] - o[i - impulse_bars + 1]
        d = np.sign(move)
        if d == 0 or abs(move) < impulse_atr * atr[i]:
            continue
        ob = -1
        for j in range(i - impulse_bars, max(i - impulse_bars - 8, 0), -1):
            if (c[j] < o[j]) if d > 0 else (c[j] > o[j]):
                ob = j; break
        if ob < 0:
            continue
        if require_imbalance and ob + 2 < len(htf):
            ok = (l[ob + 2] > h[ob]) if d > 0 else (h[ob + 2] < l[ob])
            if not ok:
                continue
        obs.append({
            "dir": int(d),
            "lo": float(min(o[ob], c[ob], l[ob])),
            "hi": float(max(o[ob], c[ob], h[ob])),
            # exploitable seulement APRES la cloture de la bougie d'impulsion
            "valid_from": htf.index[i] + pd.Timedelta(freq),
            "expires": htf.index[i] + pd.Timedelta(freq) * 20,
        })
    return obs


def backtest_scalp(m5: pd.DataFrame, cfg: ScalpConfig) -> tuple[pd.DataFrame, dict]:
    idx = m5.index
    o = m5.open.to_numpy(); h = m5.high.to_numpy()
    l = m5.low.to_numpy(); c = m5.close.to_numpy()
    hours = idx.hour.to_numpy(); days = idx.normalize(); n = len(idx)
    atr = _atr(m5, 288)
    sh, sl = _swings(h, l, cfg.fractal_k)
    if "tickvol" in m5.columns:
        tv = m5.tickvol.to_numpy(dtype=float)
        vmed = m5.tickvol.rolling(288, min_periods=100).median().to_numpy()
        vratio = np.divide(tv, vmed, out=np.full(len(tv), np.nan), where=vmed > 0)
    else:
        vratio = np.full(n, np.nan)

    h1 = m5.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    spread_h1 = (h1.close.ewm(span=cfg.htf_fast).mean()
                 - h1.close.ewm(span=cfg.htf_slow).mean())
    tr = np.sign(spread_h1).shift(1)
    trend = tr.reindex(idx, method="ffill").to_numpy()
    # force de la tendance : amplitude normalisee, pas seulement le signe
    strength = (spread_h1.abs() / h1.close.rolling(24).std()).shift(1)
    tstrength = strength.reindex(idx, method="ffill").to_numpy()
    # alignement du timeframe superieur (H4)
    h4 = m5.resample("4h").agg({"close":"last"}).dropna()
    tr4 = np.sign(h4.close.ewm(span=10).mean() - h4.close.ewm(span=30).mean()).shift(1)
    trend4 = tr4.reindex(idx, method="ffill").to_numpy()

    kz = np.zeros(n, bool)
    for a, b in cfg.killzones:
        kz |= (hours >= a) & (hours < b)

    atr_pct = pd.Series(atr, index=idx).rolling(
        2880, min_periods=500).rank(pct=True).to_numpy() * 100.0

    def confirmed(j: int, d: int) -> str | None:
        """Retourne les confirmations validees, ou None si le seuil n'est
        pas atteint. Empiler des confirmations reduit la frequence : le test
        dira si cela ameliore vraiment l'esperance ou seulement le confort."""
        found = []
        body = abs(c[j] - o[j]); rng = h[j] - l[j]
        if cfg.conf_rejection_wick and rng > 0:
            wick = (min(o[j], c[j]) - l[j]) if d > 0 else (h[j] - max(o[j], c[j]))
            if body > 0 and wick >= cfg.wick_ratio * body and \
               ((c[j] > o[j]) if d > 0 else (c[j] < o[j])):
                found.append("meche_rejet")
        if cfg.conf_engulfing and j >= 1:
            prev_bear = c[j-1] < o[j-1]; prev_bull = c[j-1] > o[j-1]
            if d > 0 and prev_bear and c[j] > o[j] and c[j] >= o[j-1] and o[j] <= c[j-1]:
                found.append("englobante")
            if d < 0 and prev_bull and c[j] < o[j] and c[j] <= o[j-1] and o[j] >= c[j-1]:
                found.append("englobante")
        if cfg.conf_choch:
            micro = np.nan
            for k in range(j - cfg.fractal_k, max(j - 12, 1), -1):
                if d > 0 and sh[k]: micro = h[k]; break
                if d < 0 and sl[k]: micro = l[k]; break
            if not np.isnan(micro) and ((c[j] > micro) if d > 0 else (c[j] < micro)):
                found.append("choch")
        if cfg.allowed_confirmations:
            found = [f for f in found if f in cfg.allowed_confirmations]
        if len(found) < cfg.min_confirmations:
            return None
        return "+".join(sorted(found))

    htf_obs = []
    if cfg.ob_timeframe != "5min":
        htf_obs = detect_htf_obs(m5, cfg.ob_timeframe, cfg.impulse_atr,
                                 cfg.impulse_bars, cfg.require_imbalance)
        htf_valid = np.array([x["valid_from"].value for x in htf_obs])

    equity = cfg.capital; trades = []; per_day = {}
    rej = {"pas_dOB": 0, "pas_imbalance": 0, "zone_non_touchee": 0,
           "pas_de_confirmation": 0, "risque_trop_grand": 0}
    i = 300
    while i < n - cfg.poi_valid_bars - cfg.max_hold_bars - 2:
        day = days[i]
        if not kz[i] or np.isnan(atr[i]) or atr[i] <= 0:
            i += 1; continue
        if per_day.get(day, 0) >= cfg.max_trades_per_day:
            i += 1; continue

        if cfg.allowed_hours and hours[i] not in cfg.allowed_hours:
            i += 1; continue
        pv = atr_pct[i]
        if not np.isnan(pv) and (pv < cfg.atr_pct_floor or pv > cfg.atr_pct_cap):
            i += 1; continue

        d = trend[i] if cfg.use_trend_filter else np.sign(c[i] - o[i - cfg.impulse_bars])
        if np.isnan(d) or d == 0:
            i += 1; continue
        if cfg.direction_filter == "long" and d < 0: i += 1; continue
        if cfg.direction_filter == "short" and d > 0: i += 1; continue
        if cfg.min_trend_force > 0:
            tf = tstrength[i]
            if np.isnan(tf) or tf < cfg.min_trend_force: i += 1; continue
        if cfg.require_h4:
            t4 = trend4[i]
            if np.isnan(t4) or t4 != d: i += 1; continue

        move = c[i] - o[i - cfg.impulse_bars + 1]
        if np.sign(move) != d or abs(move) < cfg.impulse_atr * atr[i]:
            i += 1; continue
        if cfg.min_impulse_vol > 0:
            seg = vratio[max(i - cfg.impulse_bars + 1, 0):i + 1]
            if np.all(np.isnan(seg)) or np.nanmax(seg) < cfg.min_impulse_vol:
                i += 1; continue

        if cfg.ob_timeframe != "5min":
            # Order Block issu de l'unite superieure, deja valide avant
            # l'instant courant. Le prix doit revenir dedans maintenant.
            ts = idx[i].value
            cand = [x for k, x in enumerate(htf_obs)
                    if htf_valid[k] <= ts and x["dir"] == d
                    and x["expires"].value >= ts
                    and x["lo"] <= h[i] and x["hi"] >= l[i]]
            if not cand:
                rej["pas_dOB"] += 1; i += 1; continue
            zone = cand[-1]
            ob, ob_lo, ob_hi = i, zone["lo"], zone["hi"]
        else:
            ob = -1
            for j in range(i - cfg.impulse_bars, max(i - cfg.impulse_bars - 10, 1), -1):
                if (c[j] < o[j]) if d > 0 else (c[j] > o[j]):
                    ob = j; break
            if ob < 0:
                rej["pas_dOB"] += 1; i += 1; continue
            ob_lo, ob_hi = min(o[ob], c[ob], l[ob]), max(o[ob], c[ob], h[ob])

        if cfg.require_imbalance and cfg.ob_timeframe == "5min":
            ok = (l[ob+2] > h[ob]) if d > 0 else (h[ob+2] < l[ob])
            if ob + 2 >= n or not ok:
                rej["pas_imbalance"] += 1; i += 1; continue

        # fraicheur de l'Order Block : combien de fois a-t-il deja ete
        # touche depuis sa formation ? Un OB "frais" est cense mieux tenir.
        prior_touches = 0
        for j in range(min(ob + 1, i), i + 1):
            if l[j] <= ob_hi and h[j] >= ob_lo:
                prior_touches += 1

        # retour dans l'Order Block (zone OTE non requise)
        touch = -1
        for j in range(i + 1, min(i + cfg.poi_valid_bars, n)):
            if (l[j] < ob_lo) if d > 0 else (h[j] > ob_hi):
                break
            if l[j] <= ob_hi and h[j] >= ob_lo:
                touch = j; break
        if touch < 0:
            rej["zone_non_touchee"] += 1; i += 1; continue

        # UNE confirmation suffit
        entry_at, kind = -1, None
        for j in range(touch, min(touch + cfg.confirm_bars, n)):
            k = confirmed(j, int(d))
            if k and cfg.min_vol_ratio > 0:
                vr = vratio[j]
                if np.isnan(vr) or vr < cfg.min_vol_ratio:
                    k = None          # confirmation sans volume = ignoree
            if k: entry_at, kind = j, k; break
            if (l[j] < ob_lo) if d > 0 else (h[j] > ob_hi): break
        if entry_at < 0:
            rej["pas_de_confirmation"] += 1; i = touch + 1; continue

        entry_px = c[entry_at]
        if cfg.stop_mode == "confirmation":
            base = l[entry_at] if d > 0 else h[entry_at]
        else:
            base = ob_lo if d > 0 else ob_hi
        stop_px = base - d * cfg.stop_buffer_atr * atr[entry_at]
        risk = abs(entry_px - stop_px)
        if risk <= 0:
            i = entry_at + 1; continue
        if risk * cfg.units + (cfg.spread_usd + cfg.slippage_usd) * cfg.units \
                > equity * cfg.max_risk_pct:
            rej["risque_trop_grand"] += 1; i = entry_at + 1; continue
        target_px = entry_px + d * cfg.tp_r * risk

        r_mult, why = 0.0, "fin_fenetre"
        partial_done = False   # defini avant toute sortie possible
        if (l[entry_at] <= stop_px) if d > 0 else (h[entry_at] >= stop_px):
            r_mult, why = -1.0, "stop"
        else:
            hit = False
            be_px = entry_px + d * (cfg.spread_usd + cfg.slippage_usd)
            be_done = False
            # --- TP1 partiel ----------------------------------------------
            if cfg.partial_r > 0:
                p_dist = cfg.partial_r * risk
            elif cfg.partial_usd > 0:
                p_dist = cfg.partial_usd
            else:
                p_dist = 0.0
            partial_px = entry_px + d * p_dist
            remaining, realised = 1.0, 0.0
            if p_dist <= 0:
                remaining = 1.0

            for j in range(entry_at + 1, min(entry_at + cfg.max_hold_bars, n)):
                s_ = (l[j] <= stop_px) if d > 0 else (h[j] >= stop_px)
                t_ = (h[j] >= target_px) if d > 0 else (l[j] <= target_px)
                # ordre conservateur dans une meme bougie : stop, puis TP1,
                # puis TP2. On ne suppose jamais la sequence favorable.
                if s_:
                    realised += remaining * (stop_px - entry_px) * d / risk
                    r_mult, hit = realised, True
                    why = ("breakeven" if be_done else "stop")
                    break
                if p_dist > 0 and not partial_done:
                    reached = (h[j] >= partial_px) if d > 0 else (l[j] <= partial_px)
                    if reached:
                        realised += cfg.partial_fraction * p_dist / risk
                        remaining -= cfg.partial_fraction
                        partial_done = True
                        if cfg.be_after_partial:
                            stop_px, be_done = be_px, True
                if t_:
                    realised += remaining * cfg.tp_r
                    r_mult, why, hit = realised, "cible", True; break
                if cfg.breakeven_at_r > 0 and not be_done:
                    reached = (h[j] >= entry_px + d * cfg.breakeven_at_r * risk) if d > 0 \
                        else (l[j] <= entry_px + d * cfg.breakeven_at_r * risk)
                    if reached:
                        stop_px, be_done = be_px, True
            if not hit:
                j = min(entry_at + cfg.max_hold_bars - 1, n - 1)
                r_mult = realised + remaining * (c[j] - entry_px) * d / risk

        cost = (cfg.spread_usd + cfg.slippage_usd) * cfg.units
        pnl = r_mult * risk * cfg.units - cost
        equity += pnl
        per_day[day] = per_day.get(day, 0) + 1
        trades.append({"date": day, "entry_ts": idx[entry_at],
                       "direction": "long" if d > 0 else "short",
                       "confirmation": kind, "stop_dist": risk,
                       "R": pnl / (risk * cfg.units), "pnl": pnl,
                       "hour": idx[entry_at].hour, "partiel": partial_done,
                       "ob_size_atr": (ob_hi - ob_lo) / atr[i],
                       "poi_age": touch - i,
                       "impulse_atr": abs(move) / atr[i],
                       "trend_force": tstrength[i],
                       "h4_aligne": bool(trend4[i] == d) if not np.isnan(trend4[i]) else None,
                       "touches_ob": prior_touches,
                       "jour_semaine": idx[entry_at].dayofweek,
                       "vol_ratio": vratio[entry_at],
                       "exit": why, "equity": equity})
        if equity < 50: break
        i = entry_at + 1

    df = pd.DataFrame(trades); nd = idx.normalize().nunique()
    st = {"jours": nd, "nb_trades": len(df),
          "trades_par_jour": len(df) / nd if nd else 0.0,
          "capital_final": equity, "rejets": rej}
    if not df.empty:
        st |= {"winrate": (df.pnl > 0).mean(), "R_moyen": df.R.mean(),
               "stop_median": df.stop_dist.median(),
               "part_cibles": (df.exit == "cible").mean()}
    return df, st
