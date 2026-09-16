"""Modele Order Block + Imbalance + OTE, multi-timeframe (H1 -> M5 -> M5 conf).

Specification exacte demandee :
  - contexte de tendance en H1, on ne trade que DANS le sens de la tendance
  - Order Block : derniere bougie opposee avant le mouvement impulsif
  - Imbalance obligatoire : l'impulsion doit laisser un FVG (3 bougies)
  - zone d'entree : OTE (0.618-0.786) OU retracement 0.5, du leg impulsif
  - liquidite : cible sur le pool oppose (extremes veille / session)
  - confirmation en unite de temps inferieure avant d'entrer (micro-CHoCH)

Differences avec le modele ICT deja teste (quant/smc.py) :
  le point d'interet est l'Order Block et non le FVG seul, l'entree est
  conditionnee a une zone de Fibonacci, et surtout l'entree n'est plus un
  ordre limite aveugle mais exige une confirmation. C'est precisement ce
  troisieme point qui merite d'etre mesure : la confirmation ajoute-t-elle
  de l'information, ou fait-elle seulement entrer plus tard et plus cher ?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OBConfig:
    capital: float = 450.0
    units: float = 1.0
    spread_usd: float = 0.25
    slippage_usd: float = 0.10
    max_risk_pct: float = 0.99          # 0.99 = contrainte de compte desactivee

    htf_fast: int = 20                  # EMA rapide H1 (tendance)
    htf_slow: int = 60
    fractal_k: int = 2
    impulse_atr: float = 1.2            # leg impulsif >= k x ATR(M5)
    require_imbalance: bool = True
    entry_zone: str = "ote"             # "ote" | "fib50" | "ob" (zone OB brute)
    ote_low: float = 0.618
    ote_high: float = 0.786
    fib50_tol: float = 0.06             # 0.5 +/- tolerance
    require_confirmation: bool = True   # micro-CHoCH avant entree
    confirm_bars: int = 12              # fenetre de confirmation
    poi_valid_bars: int = 72            # duree de vie du POI (6h en M5)
    stop_buffer_atr: float = 0.25
    min_rr: float = 1.5
    max_rr: float = 4.0
    exogenous_stop_atr: float = 0.0     # >0 : stop fixe en ATR, pour isoler
                                        # l'effet du placement du stop sur OB
    max_trades_per_day: int = 3
    use_killzone: bool = True
    killzones: tuple[tuple[int, int], ...] = ((7, 11), (12, 16))


def _atr(df: pd.DataFrame, window: int) -> np.ndarray:
    prev = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(),
                    (df.low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window // 2).mean().to_numpy()


def _swings(h: np.ndarray, l: np.ndarray, k: int):
    n = len(h)
    sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(k, n - k):
        if h[i] == h[i - k:i + k + 1].max() and (h[i - k:i + k + 1] == h[i]).sum() == 1:
            sh[i] = True
        if l[i] == l[i - k:i + k + 1].min() and (l[i - k:i + k + 1] == l[i]).sum() == 1:
            sl[i] = True
    return sh, sl


def backtest_ob(m5: pd.DataFrame, cfg: OBConfig) -> tuple[pd.DataFrame, dict]:
    idx = m5.index
    o = m5.open.to_numpy(); h = m5.high.to_numpy()
    l = m5.low.to_numpy(); c = m5.close.to_numpy()
    hours = idx.hour.to_numpy()
    days = idx.normalize()
    n = len(idx)

    atr5 = _atr(m5, 288)                      # ATR M5 sur 24h

    # ---- contexte H1 : tendance, decalee d'une bougie (pas de lookahead) --
    h1 = m5.resample("1h").agg({"open": "first", "high": "max",
                                "low": "min", "close": "last"}).dropna()
    trend_h1 = np.sign(h1.close.ewm(span=cfg.htf_fast).mean()
                       - h1.close.ewm(span=cfg.htf_slow).mean()).shift(1)
    trend = trend_h1.reindex(idx, method="ffill").to_numpy()

    # ---- pools de liquidite : extremes de la veille -----------------------
    dk = pd.Index(days)
    dfd = pd.DataFrame({"h": h, "l": l}, index=idx)
    prev_hi = dfd.groupby(dk).h.max().shift(1)
    prev_lo = dfd.groupby(dk).l.min().shift(1)
    asia = (hours >= 0) & (hours < 6)
    asia_hi = dfd[asia].groupby(dk[asia]).h.max()
    asia_lo = dfd[asia].groupby(dk[asia]).l.min()

    in_kz = np.zeros(n, bool)
    for a, b in cfg.killzones:
        in_kz |= (hours >= a) & (hours < b)
    if not cfg.use_killzone:
        in_kz[:] = True

    sh, sl = _swings(h, l, cfg.fractal_k)

    equity = cfg.capital
    trades = []
    rej = {"pas_impulsion": 0, "pas_imbalance": 0, "hors_tendance": 0,
           "zone_non_atteinte": 0, "pas_de_confirmation": 0,
           "RR_insuffisant": 0, "risque_trop_grand": 0}
    per_day: dict = {}

    i = 300
    while i < n - cfg.poi_valid_bars - cfg.confirm_bars - 2:
        day = days[i]
        if not in_kz[i] or np.isnan(atr5[i]) or atr5[i] <= 0:
            i += 1; continue
        if per_day.get(day, 0) >= cfg.max_trades_per_day:
            i += 1; continue
        bias = trend[i]
        if np.isnan(bias) or bias == 0:
            i += 1; continue

        # ---- 1. LEG IMPULSIF dans le sens de la tendance H1 --------------
        # mesure sur 6 bougies M5 (30 min) : c'est l'echelle d'une impulsion
        lookf = 6
        seg_move = c[i] - o[i - lookf + 1]
        if np.sign(seg_move) != bias or abs(seg_move) < cfg.impulse_atr * atr5[i]:
            i += 1; continue

        # ---- 2. ORDER BLOCK : derniere bougie opposee avant l'impulsion ---
        ob = -1
        for j in range(i - lookf, max(i - lookf - 12, 1), -1):
            opposite = (c[j] < o[j]) if bias > 0 else (c[j] > o[j])
            if opposite:
                ob = j; break
        if ob < 0:
            rej["pas_impulsion"] += 1; i += 1; continue
        ob_lo, ob_hi = min(o[ob], c[ob], l[ob]), max(o[ob], c[ob], h[ob])

        # ---- 3. IMBALANCE laissee par l'impulsion -------------------------
        has_fvg = False
        for j in range(ob + 1, min(i, n - 1)):
            if bias > 0 and l[j + 1] > h[j - 1]:
                has_fvg = True; break
            if bias < 0 and h[j + 1] < l[j - 1]:
                has_fvg = True; break
        if cfg.require_imbalance and not has_fvg:
            rej["pas_imbalance"] += 1; i += 1; continue

        # ---- 4. ZONE D'ENTREE : OTE ou 0.5 du leg impulsif ----------------
        leg_lo = min(l[ob:i + 1]); leg_hi = max(h[ob:i + 1])
        leg = leg_hi - leg_lo
        if leg <= 0:
            i += 1; continue
        if cfg.entry_zone == "ote":
            if bias > 0:
                z_hi = leg_hi - cfg.ote_low * leg
                z_lo = leg_hi - cfg.ote_high * leg
            else:
                z_lo = leg_lo + cfg.ote_low * leg
                z_hi = leg_lo + cfg.ote_high * leg
        elif cfg.entry_zone == "fib50":
            mid = leg_hi - 0.5 * leg if bias > 0 else leg_lo + 0.5 * leg
            z_lo, z_hi = mid - cfg.fib50_tol * leg, mid + cfg.fib50_tol * leg
        else:
            z_lo, z_hi = ob_lo, ob_hi
        # intersection avec l'Order Block : les deux doivent se recouvrir
        zone_lo, zone_hi = max(z_lo, ob_lo), min(z_hi, ob_hi)
        if zone_hi <= zone_lo:
            rej["zone_non_atteinte"] += 1; i += 1; continue

        # ---- 5. attente du retour dans la zone ---------------------------
        touch = -1
        for j in range(i + 1, min(i + cfg.poi_valid_bars, n)):
            # l'invalidation PRIME sur le touche : une bougie qui traverse la
            # zone et casse l'Order Block dans le meme mouvement n'est pas une
            # entree, c'est un stop. L'ordre inverse offrait une option gratuite
            # sur exactement les bougies qui auraient du nous sortir.
            invalid = (l[j] < ob_lo) if bias > 0 else (h[j] > ob_hi)
            if invalid:
                break
            if l[j] <= zone_hi and h[j] >= zone_lo:
                touch = j; break
        if touch < 0:
            rej["zone_non_atteinte"] += 1; i += 1; continue

        # ---- 6. CONFIRMATION en unite de temps inferieure -----------------
        entry_at, entry_px = -1, np.nan
        if cfg.require_confirmation:
            # micro-CHoCH : cloture au-dela du dernier micro-swing oppose
            micro = np.nan
            # on n'utilise que les swings DEJA CONFIRMES a l'instant touch :
            # une fractale d'indice j n'est connue qu'en j + k.
            for j in range(touch - cfg.fractal_k, max(touch - 10, 1), -1):
                if bias > 0 and sh[j]:
                    micro = h[j]; break
                if bias < 0 and sl[j]:
                    micro = l[j]; break
            if np.isnan(micro):
                micro = h[touch] if bias > 0 else l[touch]
            for j in range(touch + 1, min(touch + cfg.confirm_bars, n)):
                confirmed = (c[j] > micro) if bias > 0 else (c[j] < micro)
                if confirmed:
                    entry_at, entry_px = j, c[j]; break
                broke = (l[j] < ob_lo) if bias > 0 else (h[j] > ob_hi)
                if broke:
                    break
            if entry_at < 0:
                rej["pas_de_confirmation"] += 1; i = touch + 1; continue
        else:
            entry_at = touch
            entry_px = min(max(c[touch], zone_lo), zone_hi)

        # ---- 7. stop / cible ---------------------------------------------
        if cfg.exogenous_stop_atr > 0:
            risk = cfg.exogenous_stop_atr * atr5[entry_at]
            stop_px = entry_px - bias * risk
        else:
            stop_px = (ob_lo - cfg.stop_buffer_atr * atr5[entry_at]) if bias > 0 \
                else (ob_hi + cfg.stop_buffer_atr * atr5[entry_at])
            risk = abs(entry_px - stop_px)
        if risk <= 0:
            i = entry_at + 1; continue

        pools = []
        for src in ((prev_hi, asia_hi) if bias > 0 else (prev_lo, asia_lo)):
            v = src.get(day, np.nan)
            if not np.isnan(v) and ((v > entry_px) if bias > 0 else (v < entry_px)):
                pools.append(v)
        if pools:
            tgt = min(pools) if bias > 0 else max(pools)
            rr = abs(tgt - entry_px) / risk
        else:
            rr = cfg.max_rr
        if rr < cfg.min_rr:
            rej["RR_insuffisant"] += 1; i = entry_at + 1; continue
        rr = min(rr, cfg.max_rr)
        target_px = entry_px + bias * rr * risk

        risk_usd = risk * cfg.units + (cfg.spread_usd + cfg.slippage_usd) * cfg.units
        if risk_usd > equity * cfg.max_risk_pct:
            rej["risque_trop_grand"] += 1; i = entry_at + 1; continue

        # ---- simulation ---------------------------------------------------
        # Traitement ASYMETRIQUE et volontairement conservateur de la bougie
        # d'entree : on y teste le stop, jamais la cible. L'extreme favorable
        # de cette bougie peut avoir eu lieu AVANT notre remplissage - le
        # compter comme un gain est un lookahage intra-bougie, et c'est
        # exactement lui qui rendait le modele rentable sur du bruit pur.
        r_mult, exit_reason = 0.0, "fin_fenetre"
        first_stop = (l[entry_at] <= stop_px) if bias > 0 else (h[entry_at] >= stop_px)
        if first_stop:
            r_mult, exit_reason = -1.0, "stop"
        else:
          for j in range(entry_at + 1, min(entry_at + 288, n)):
            hit_s = (l[j] <= stop_px) if bias > 0 else (h[j] >= stop_px)
            hit_t = (h[j] >= target_px) if bias > 0 else (l[j] <= target_px)
            if hit_s:
                r_mult, exit_reason = -1.0, "stop"; break
            if hit_t:
                r_mult, exit_reason = rr, "cible"; break
          else:
            j = min(entry_at + 287, n - 1)
            r_mult = (c[j] - entry_px) * bias / risk

        cost = (cfg.spread_usd + cfg.slippage_usd) * cfg.units
        pnl = r_mult * risk * cfg.units - cost
        equity += pnl
        per_day[day] = per_day.get(day, 0) + 1
        trades.append({"date": day, "entry_ts": idx[entry_at],
                       "direction": "long" if bias > 0 else "short",
                       "rr_cible": rr, "stop_dist": risk,
                       "R": pnl / (risk * cfg.units), "pnl": pnl,
                       "exit": exit_reason, "equity": equity})
        if equity < 50:
            break
        i = entry_at + 1

    df = pd.DataFrame(trades)
    nd = idx.normalize().nunique()
    stats = {"jours": nd, "nb_trades": len(df),
             "trades_par_jour": len(df) / nd if nd else 0.0,
             "capital_final": equity, "rejets": rej}
    if not df.empty:
        stats |= {"winrate": (df.pnl > 0).mean(), "R_moyen": df.R.mean(),
                  "rr_median": df.rr_cible.median(),
                  "stop_median": df.stop_dist.median(),
                  "part_cibles": (df.exit == "cible").mean()}
    return df, stats
