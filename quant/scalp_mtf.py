"""Scalping Order Block avec confirmation en unite de temps inferieure.

Architecture conforme a la pratique :
  - Order Block, impulsion et zone  -> detectes en M5
  - retour dans la zone             -> surveille en M1
  - confirmation d'entree           -> cherchee en M1
  - stop                            -> bas de l'Order Block M5 (invalidation)
  - gestion                         -> suivie en M1

L'interet attendu de confirmer en M1 plutot qu'en M5 est mecanique : on
entre plus tot dans le mouvement, donc plus pres du bas de l'Order Block,
donc le risque par trade est plus petit. A mouvement egal, le R est plus
grand. Le risque symetrique est d'entrer sur des signaux plus bruites.

Meme discipline anti-lookahead : entree a la CLOTURE de la bougie M1 de
confirmation, gestion a partir de la bougie suivante, stop seul teste sur
la bougie d'entree, fractales lues avec leur retard de confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MTFConfig:
    capital: float = 450.0
    units: float = 1.0
    # Spread MESURE sur la colonne <SPREAD> de l'export MT5 du courtier :
    # 588 732 minutes, mediane 0.150 USD l'once, q95 0.200. La valeur de
    # 0.23 utilisee auparavant etait une hypothese pessimiste.
    # Deux heures serveur font exception, 23h et 01h, ou le rollover porte
    # 9 a 10% des minutes au-dela de 0.30. La plage horaire les exclut.
    spread_usd: float = 0.15
    # Le slippage reste une hypothese : il ne se lit pas dans des donnees
    # de bougies. A confirmer sur releve de compte en forward test.
    slippage_usd: float = 0.10

    htf_fast: int = 20
    htf_slow: int = 60
    min_trend_force: float = 1.0
    fractal_k: int = 2

    impulse_bars: int = 5             # en bougies M5
    impulse_atr: float = 0.8

    # Mode de detection de l'Order Block.
    #   "fenetre"  : recherche dans une fenetre fixe situee cinq a quinze
    #                bougies en arriere. C'est la version qui a produit les
    #                resultats publies, zone prise meches comprises.
    #   "remontee" : on part de la bougie courante et l'on remonte jusqu'a
    #                la premiere bougie de couleur opposee, en exigeant que
    #                l'impulsion ait depasse cette bougie. Definition
    #                litterale, zone prise au corps.
    # Fraction de recouvrement au-dela de laquelle une nouvelle zone serait
    # ecartee comme doublon de la precedente. LAISSER A 1.0 : la mesure sur
    # 436 jours montre que la deduplication fait tomber l'ecart au temoin de
    # +0.050 a +0.012 R. Les zones qui se recouvrent ne sont donc pas de
    # simples doublons, et les retirer coute de l'information. Le parametre
    # reste expose pour que le resultat puisse etre reproduit.
    dedup_overlap: float = 1.0
    ob_mode: str = "fenetre"
    ob_lookback: int = 20
    min_leg_bars: int = 2
    use_body: bool = False

    # Types de confirmation ACTIFS. Les mesurer en isolation plutot que par
    # elimination : la fonction s'arretant a la premiere trouvee, un type
    # place en dernier n'etait vu que lorsque les autres manquaient, ce qui
    # ne compare pas des signaux mais des restes.
    #   meche_rejet   meche >= wick_ratio x corps, cloture du bon sens
    #   englobante    la bougie englobe la precedente, de couleur opposee
    #   choch         cloture au-dela du dernier extreme de structure oppose
    #   pinbar        variante stricte de la meche : >= 2.5 x corps
    #   inside_break  sortie d'une bougie interieure a la precedente
    confirm_types: tuple[str, ...] = ("meche_rejet", "englobante", "choch")
    # A False, on entre des le TOUCHER de la zone, sans rien attendre.
    # C'est le cas de base : si confirmer n'ameliore pas ce resultat,
    # l'appareillage de confirmations ne sert a rien.
    require_confirm: bool = True

    confirm_tf: str = "1min"          # "1min" | "3min" | "5min"
    confirm_bars: int = 30            # fenetre de confirmation, en bougies M1
    poi_valid_bars: int = 48          # duree de vie de l'OB, en bougies M5
    wick_ratio: float = 1.5

    stop_buffer_atr: float = 0.15
    tp_r: float = 3.0
    partial_r: float = 1.0
    partial_fraction: float = 0.33
    max_hold_min: int = 240           # 4h
    max_trades_per_day: int = 10
    killzones: tuple[tuple[int, int], ...] = ((6, 20),)


def _atr(df, w):
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


def backtest_mtf(m5: pd.DataFrame, m1: pd.DataFrame,
                 cfg: MTFConfig) -> tuple[pd.DataFrame, dict]:
    if cfg.confirm_tf != "1min":
        m1 = m1.resample(cfg.confirm_tf).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna(subset=["open", "close"])

    # --- contexte M5 -------------------------------------------------------
    o5 = m5.open.to_numpy(); h5 = m5.high.to_numpy()
    l5 = m5.low.to_numpy(); c5 = m5.close.to_numpy()
    atr5 = _atr(m5, 288)
    i5 = m5.index
    hours5 = i5.hour.to_numpy()

    h1b = m5.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    spread = h1b.close.ewm(span=cfg.htf_fast).mean() - h1b.close.ewm(span=cfg.htf_slow).mean()
    trend = np.sign(spread).shift(1).reindex(i5, method="ffill").to_numpy()
    force = (spread.abs() / h1b.close.rolling(24).std()).shift(1).reindex(i5, method="ffill").to_numpy()

    kz = np.zeros(len(i5), bool)
    for a, b in cfg.killzones:
        kz |= (hours5 >= a) & (hours5 < b)

    # --- unite de confirmation --------------------------------------------
    o1 = m1.open.to_numpy(); hh = m1.high.to_numpy()
    ll = m1.low.to_numpy(); cc = m1.close.to_numpy()
    i1 = m1.index
    sh1, sl1 = _swings(hh, ll, cfg.fractal_k)
    pos1 = i1.values.astype("datetime64[ns]")

    actifs = set(cfg.confirm_types)

    def confirmed(j: int, d: int) -> str | None:
        body = abs(cc[j] - o1[j]); rng = hh[j] - ll[j]
        wick = (min(o1[j], cc[j]) - ll[j]) if d > 0 else (hh[j] - max(o1[j], cc[j]))
        sens = (cc[j] > o1[j]) if d > 0 else (cc[j] < o1[j])
        if "pinbar" in actifs and rng > 0 and body > 0 and sens:
            # Variante stricte de la meche : le corps doit en outre se
            # trouver dans le tiers oppose de la bougie, faute de quoi une
            # longue bougie a deux meches passerait pour un rejet.
            tiers = rng / 3.0
            place = (min(o1[j], cc[j]) - ll[j] >= 2 * tiers) if d > 0 \
                else (hh[j] - max(o1[j], cc[j]) >= 2 * tiers)
            if wick >= 2.5 * body and place:
                return "pinbar"
        if "inside_break" in actifs and j >= 2:
            # La bougie j-1 est contenue dans la j-2, et la bougie j en sort
            # du bon cote : compression puis expansion.
            dedans = hh[j-1] <= hh[j-2] and ll[j-1] >= ll[j-2]
            if dedans and ((cc[j] > hh[j-1]) if d > 0 else (cc[j] < ll[j-1])):
                return "inside_break"
        if "meche_rejet" in actifs and rng > 0 and body > 0:
            if wick >= cfg.wick_ratio * body and sens:
                return "meche_rejet"
        if "englobante" in actifs and j >= 1:
            if d > 0 and cc[j-1] < o1[j-1] and cc[j] > o1[j] and cc[j] >= o1[j-1] and o1[j] <= cc[j-1]:
                return "englobante"
            if d < 0 and cc[j-1] > o1[j-1] and cc[j] < o1[j] and cc[j] <= o1[j-1] and o1[j] >= cc[j-1]:
                return "englobante"
        if "choch" not in actifs:
            return None
        micro = np.nan
        for k in range(j - cfg.fractal_k, max(j - 20, 1), -1):
            if d > 0 and sh1[k]: micro = hh[k]; break
            if d < 0 and sl1[k]: micro = ll[k]; break
        if not np.isnan(micro) and ((cc[j] > micro) if d > 0 else (cc[j] < micro)):
            return "choch"
        return None

    equity = cfg.capital; trades = []; per_day = {}
    zones_recentes: list = []
    rej = {"pas_dOB": 0, "zone_non_touchee": 0, "pas_de_confirmation": 0}
    n5 = len(i5); n1 = len(i1)
    i = 300
    while i < n5 - cfg.poi_valid_bars - 2:
        day = i5[i].normalize()
        if not kz[i] or np.isnan(atr5[i]) or atr5[i] <= 0:
            i += 1; continue
        if per_day.get(day, 0) >= cfg.max_trades_per_day:
            i += 1; continue
        d = trend[i]
        if np.isnan(d) or d == 0: i += 1; continue
        if cfg.min_trend_force > 0 and (np.isnan(force[i]) or force[i] < cfg.min_trend_force):
            i += 1; continue

        move = c5[i] - o5[i - cfg.impulse_bars + 1]
        if np.sign(move) != d or abs(move) < cfg.impulse_atr * atr5[i]:
            i += 1; continue

        ob = -1
        if cfg.ob_mode == "remontee":
            # Definition litterale : premiere bougie de couleur opposee en
            # remontant depuis la bougie courante.
            for j in range(i - 1, max(i - cfg.ob_lookback, 1), -1):
                if (c5[j] < o5[j]) if d > 0 else (c5[j] > o5[j]):
                    ob = j; break
            if ob >= 0:
                # L'impulsion doit avoir DEPASSE l'Order Block, sans quoi il
                # s'agit d'une oscillation interne a une consolidation.
                broke = (c5[i] > h5[ob]) if d > 0 else (c5[i] < l5[ob])
                amp = abs(c5[i] - (l5[ob] if d > 0 else h5[ob]))
                if (i - ob - 1) < cfg.min_leg_bars or not broke \
                        or amp < cfg.impulse_atr * atr5[i]:
                    ob = -1
        else:
            for j in range(i - cfg.impulse_bars, max(i - cfg.impulse_bars - 10, 1), -1):
                if (c5[j] < o5[j]) if d > 0 else (c5[j] > o5[j]):
                    ob = j; break
        if ob < 0: rej["pas_dOB"] += 1; i += 1; continue
        if cfg.use_body:
            ob_lo = min(o5[ob], c5[ob]); ob_hi = max(o5[ob], c5[ob])
        else:
            ob_lo = min(o5[ob], c5[ob], l5[ob]); ob_hi = max(o5[ob], c5[ob], h5[ob])

        # En mode fenetre, la condition d'impulsion reste vraie pendant tout
        # le rallye : la fenetre glisse d'une bougie a chaque fois, designe
        # une bougie source voisine et produit une zone qui recouvre presque
        # la precedente. On ecarte ces quasi-doublons lorsque le seuil le
        # demande. Contrairement au plafond d'affichage de l'indicateur,
        # ceci agit sur la DETECTION et change donc le nombre de trades.
        if cfg.dedup_overlap < 1.0:
            zones_recentes = [z for z in zones_recentes
                              if i - z[0] <= cfg.poi_valid_bars]
            double = False
            for _, zd, zlo, zhi in zones_recentes:
                if zd != d:
                    continue
                inter = min(zhi, ob_hi) - max(zlo, ob_lo)
                petit = min(zhi - zlo, ob_hi - ob_lo)
                if petit > 0 and inter / petit >= cfg.dedup_overlap:
                    double = True
                    break
            if double:
                rej["doublon"] = rej.get("doublon", 0) + 1
                i += 1
                continue
            zones_recentes.append((i, d, ob_lo, ob_hi))

        # Imbalance ANCREE sur l'Order Block : la troisieme bougie du
        # mouvement ne revient pas combler l'extremite de l'OB. C'est la
        # definition retenue par l'utilisateur, et celle que l'indicateur
        # utilise pour distinguer OB+ et OB simple.
        # Elle est ENREGISTREE et non filtree : le but est de mesurer si la
        # distinction porte de l'information, pas de la supposer.
        ob_plus = False
        if ob + 2 < n5:
            ob_plus = bool((l5[ob + 2] > h5[ob]) if d > 0
                           else (h5[ob + 2] < l5[ob]))

        # --- retour dans la zone, surveille en M1 --------------------------
        start = np.searchsorted(pos1, i5[i].to_datetime64(), side="right")
        end = min(start + cfg.poi_valid_bars * 5, n1)
        touch = -1
        for j in range(start, end):
            if (ll[j] < ob_lo) if d > 0 else (hh[j] > ob_hi): break
            if ll[j] <= ob_hi and hh[j] >= ob_lo: touch = j; break
        if touch < 0: rej["zone_non_touchee"] += 1; i += 1; continue

        # --- confirmation en M1 -------------------------------------------
        entry_at, kind = -1, None
        if not cfg.require_confirm:
            entry_at, kind = touch, "aucune"
        for j in range(touch, min(touch + cfg.confirm_bars, n1)) \
                if cfg.require_confirm else []:
            k = confirmed(j, int(d))
            if k: entry_at, kind = j, k; break
            if (ll[j] < ob_lo) if d > 0 else (hh[j] > ob_hi): break
        if entry_at < 0:
            rej["pas_de_confirmation"] += 1; i += 1; continue

        entry_px = cc[entry_at]
        base = ob_lo if d > 0 else ob_hi
        stop_px = base - d * cfg.stop_buffer_atr * atr5[i]
        risk = abs(entry_px - stop_px)
        if risk <= 0: i += 1; continue
        target_px = entry_px + d * cfg.tp_r * risk
        partial_px = entry_px + d * cfg.partial_r * risk
        be_px = entry_px + d * (cfg.spread_usd + cfg.slippage_usd)

        r_mult, why, partial_done = 0.0, "fin_fenetre", False
        if (ll[entry_at] <= stop_px) if d > 0 else (hh[entry_at] >= stop_px):
            r_mult, why = -1.0, "stop"
        else:
            remaining, realised, hit = 1.0, 0.0, False
            for j in range(entry_at + 1, min(entry_at + cfg.max_hold_min, n1)):
                if (ll[j] <= stop_px) if d > 0 else (hh[j] >= stop_px):
                    realised += remaining * (stop_px - entry_px) * d / risk
                    r_mult, why, hit = realised, ("breakeven" if partial_done else "stop"), True
                    break
                if not partial_done:
                    if (hh[j] >= partial_px) if d > 0 else (ll[j] <= partial_px):
                        realised += cfg.partial_fraction * cfg.partial_r
                        remaining -= cfg.partial_fraction
                        partial_done = True
                        stop_px = be_px
                if (hh[j] >= target_px) if d > 0 else (ll[j] <= target_px):
                    realised += remaining * cfg.tp_r
                    r_mult, why, hit = realised, "cible", True; break
            if not hit:
                j = min(entry_at + cfg.max_hold_min - 1, n1 - 1)
                r_mult = realised + remaining * (cc[j] - entry_px) * d / risk

        cost = (cfg.spread_usd + cfg.slippage_usd) * cfg.units
        pnl = r_mult * risk * cfg.units - cost
        equity += pnl
        per_day[day] = per_day.get(day, 0) + 1
        trades.append({"entry_ts": i1[entry_at], "date": day,
                       "direction": "long" if d > 0 else "short",
                       "confirmation": kind, "stop_dist": risk,
                       "ob_plus": ob_plus,
                       "R": pnl / (risk * cfg.units), "pnl": pnl,
                       "exit": why, "equity": equity})
        if equity < 50: break
        # on repart apres la sortie, converti en index M5
        i = max(i + 1, int(np.searchsorted(i5.values.astype("datetime64[ns]"),
                                           i1[entry_at].to_datetime64(), side="right")))

    df = pd.DataFrame(trades); nd = i5.normalize().nunique()
    st = {"jours": nd, "nb_trades": len(df),
          "trades_par_jour": len(df) / nd if nd else 0.0,
          "capital_final": equity, "rejets": rej}
    if not df.empty:
        st |= {"winrate": (df.pnl > 0).mean(), "R_moyen": df.R.mean(),
               "stop_median": df.stop_dist.median()}
    return df, st
