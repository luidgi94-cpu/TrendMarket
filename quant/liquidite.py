"""Balayage de liquidite : le prix va-t-il chercher les stops avant de repartir ?

L'HYPOTHESE, ET POURQUOI ELLE MERITE D'ETRE TESTEE
Des ordres stop s'accumulent au-dela des extremes evidents d'un graphique :
plus haut de la veille, plus bas de la veille, sommets et creux recents.
Quiconque doit executer une taille importante a interet a declencher ces
stops, parce qu'ils fournissent la contrepartie dont il a besoin. Le prix
depasse donc brievement le niveau, absorbe les ordres, puis repart dans
l'autre sens.

Contrairement a l'affirmation "un Order Block est une zone institutionnelle",
qui ne se laisse ni verifier ni refuter, celle-ci est FALSIFIABLE : si le
balayage porte de l'information, un retour dans l'ancienne fourchette apres
un depassement doit produire davantage qu'un temoin ou les bougies ont ete
melangees.

DEFINITION RETENUE, VOLONTAIREMENT STRICTE
Un balayage haussier est constate lorsque, sur une meme bougie CLOTUREE :
  - le haut depasse un niveau de reference,
  - la cloture revient EN DESSOUS de ce niveau,
  - la meche au-dela du niveau represente une part minimale de la bougie.
On vend a la cloture de cette bougie. Le stop se place au-dessus de
l'extreme du balayage, augmente d'une marge en ATR. Symetriquement pour un
balayage baissier.

La condition de meche evite de confondre un balayage avec une cassure
franche suivie d'un repli ordinaire : c'est le rejet que l'on veut
mesurer, pas le simple fait que le prix soit repasse sous un niveau.

DISCIPLINE ANTI-ANTICIPATION
Toutes les decisions sont prises sur bougie close. Les niveaux de reference
sont lus avec leur decalage de confirmation : un sommet de pivot n'est
connu qu'apres ses k bougies de validation, et le plus haut de la veille
n'est utilise qu'a partir du lendemain. Sur la bougie d'entree, seul le
stop est teste, jamais la cible : sans cela on s'attribuerait des gains
qu'un ordre reel n'aurait pas obtenus.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LiqConfig:
    capital: float = 450.0
    units: float = 1.0
    spread_usd: float = 0.15          # mesure sur l'export du courtier
    slippage_usd: float = 0.10

    # --- niveaux de reference
    use_pdh_pdl: bool = True          # plus haut / plus bas de la veille
    use_pivots: bool = True           # sommets et creux recents
    pivot_k: int = 10                 # demi-fenetre du pivot, en bougies
    pivot_max_age: int = 288          # au-dela, le niveau est oublie (1 jour M5)

    # --- qualification du balayage
    min_wick_frac: float = 0.4        # part de la bougie au-dela du niveau
    max_close_back: float = 0.5       # la cloture doit revenir d'au moins
                                      # cette fraction de la meche

    # --- filtres
    use_trend: bool = False           # desactive par defaut : un balayage
                                      # est cense etre contrarien
    htf_fast: int = 20
    htf_slow: int = 60
    killzones: tuple[tuple[int, int], ...] = ((6, 20),)
    max_trades_per_day: int = 10

    # --- gestion
    stop_buffer_atr: float = 0.15
    tp_r: float = 3.0
    max_hold_bars: int = 48
    atr_window: int = 200


def _atr(df: pd.DataFrame, w: int) -> np.ndarray:
    prev = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(),
                    (df.low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(w, min_periods=w // 2).mean().to_numpy()


def _niveaux(df: pd.DataFrame, cfg: LiqConfig) -> tuple[list, list]:
    """Niveaux de liquidite au-dessus et en dessous, avec leur date de
    disponibilite. Chaque entree vaut (indice a partir duquel le niveau est
    CONNU, prix)."""
    h, l = df.high.to_numpy(), df.low.to_numpy()
    hauts: list[tuple[int, float]] = []
    bas: list[tuple[int, float]] = []

    if cfg.use_pivots:
        k = cfg.pivot_k
        for i in range(k, len(df) - k):
            fen_h = h[i - k:i + k + 1]
            fen_l = l[i - k:i + k + 1]
            if h[i] == fen_h.max() and (fen_h == h[i]).sum() == 1:
                # le pivot n'est confirme qu'apres ses k bougies suivantes
                hauts.append((i + k, h[i]))
            if l[i] == fen_l.min() and (fen_l == l[i]).sum() == 1:
                bas.append((i + k, l[i]))

    if cfg.use_pdh_pdl:
        jour = df.index.normalize()
        par_jour = df.groupby(jour).agg(hi=("high", "max"), lo=("low", "min"))
        pos = {d: i for i, d in enumerate(df.index)}
        jours = list(par_jour.index)
        for n in range(1, len(jours)):
            # le plus haut de la veille n'est connu qu'au premier instant
            # du jour suivant
            debut = df.index[jour == jours[n]]
            if len(debut) == 0:
                continue
            i0 = pos[debut[0]]
            hauts.append((i0, float(par_jour.hi.iloc[n - 1])))
            bas.append((i0, float(par_jour.lo.iloc[n - 1])))

    hauts.sort()
    bas.sort()
    return hauts, bas


def backtest_liquidite(m5: pd.DataFrame, cfg: LiqConfig
                       ) -> tuple[pd.DataFrame, dict]:
    o = m5.open.to_numpy(); h = m5.high.to_numpy()
    l = m5.low.to_numpy(); c = m5.close.to_numpy()
    idx = m5.index
    n = len(m5)
    atr = _atr(m5, cfg.atr_window)
    heures = idx.hour.to_numpy()

    kz = np.zeros(n, bool)
    for a, b in cfg.killzones:
        kz |= (heures >= a) & (heures < b)

    tendance = np.zeros(n)
    if cfg.use_trend:
        ef = m5.close.ewm(span=cfg.htf_fast * 12, adjust=False).mean()
        es = m5.close.ewm(span=cfg.htf_slow * 12, adjust=False).mean()
        tendance = np.sign((ef - es).shift(1).to_numpy())

    hauts, bas = _niveaux(m5, cfg)
    ih = ib = 0
    actifs_h: list[tuple[int, float]] = []
    actifs_b: list[tuple[int, float]] = []

    trades = []
    equity = cfg.capital
    par_jour: dict = {}
    i = cfg.atr_window
    rej = {"pas_de_niveau": 0, "meche_trop_courte": 0, "cloture_insuffisante": 0}

    while i < n - 1:
        while ih < len(hauts) and hauts[ih][0] <= i:
            actifs_h.append(hauts[ih]); ih += 1
        while ib < len(bas) and bas[ib][0] <= i:
            actifs_b.append(bas[ib]); ib += 1
        actifs_h = [x for x in actifs_h if i - x[0] <= cfg.pivot_max_age]
        actifs_b = [x for x in actifs_b if i - x[0] <= cfg.pivot_max_age]

        if np.isnan(atr[i]) or atr[i] <= 0 or not kz[i]:
            i += 1; continue
        jour = idx[i].date()
        if par_jour.get(jour, 0) >= cfg.max_trades_per_day:
            i += 1; continue

        taille = h[i] - l[i]
        if taille <= 0:
            i += 1; continue

        d = 0
        niveau = np.nan
        # --- balayage par le haut : on vend
        cands = [p for _, p in actifs_h if l[i] < p < h[i] and c[i] < p]
        if cands:
            niveau = max(cands)
            meche = h[i] - niveau
            retour = niveau - c[i]
            if meche / taille < cfg.min_wick_frac:
                rej["meche_trop_courte"] += 1
            elif retour < cfg.max_close_back * meche:
                rej["cloture_insuffisante"] += 1
            else:
                d = -1
        # --- balayage par le bas : on achete
        if d == 0:
            cands = [p for _, p in actifs_b if l[i] < p < h[i] and c[i] > p]
            if cands:
                niveau = min(cands)
                meche = niveau - l[i]
                retour = c[i] - niveau
                if meche / taille < cfg.min_wick_frac:
                    rej["meche_trop_courte"] += 1
                elif retour < cfg.max_close_back * meche:
                    rej["cloture_insuffisante"] += 1
                else:
                    d = 1
        if d == 0:
            if not cands:
                rej["pas_de_niveau"] += 1
            i += 1; continue

        if cfg.use_trend and tendance[i] != d:
            i += 1; continue

        entree = c[i]
        base = h[i] if d < 0 else l[i]
        # stop place AU-DELA de l'extreme du balayage : au-dessus du haut
        # pour une vente, en dessous du bas pour un achat
        stop = base - d * cfg.stop_buffer_atr * atr[i]
        risque = abs(entree - stop)
        if risque <= 0:
            i += 1; continue
        cible = entree + d * cfg.tp_r * risque

        # sur la bougie d'entree, SEUL le stop est teste
        r_mult, motif = 0.0, "fin_fenetre"
        touche = False
        if (l[i] <= stop) if d > 0 else (h[i] >= stop):
            r_mult, motif, touche = -1.0, "stop", True
        else:
            for j in range(i + 1, min(i + cfg.max_hold_bars, n)):
                if (l[j] <= stop) if d > 0 else (h[j] >= stop):
                    r_mult, motif, touche = -1.0, "stop", True; break
                if (h[j] >= cible) if d > 0 else (l[j] <= cible):
                    r_mult, motif, touche = cfg.tp_r, "cible", True; break
            if not touche:
                j = min(i + cfg.max_hold_bars - 1, n - 1)
                r_mult = (c[j] - entree) * d / risque

        cout = (cfg.spread_usd + cfg.slippage_usd) * cfg.units
        pnl = r_mult * risque * cfg.units - cout
        equity += pnl
        par_jour[jour] = par_jour.get(jour, 0) + 1
        trades.append({"ts": idx[i], "date": jour,
                       "sens": "long" if d > 0 else "short",
                       "niveau": niveau, "stop_dist": risque,
                       "R": pnl / (risque * cfg.units), "pnl": pnl,
                       "sortie": motif, "equity": equity})
        i += 1

    df = pd.DataFrame(trades)
    nj = idx.normalize().nunique()
    st = {"jours": nj, "nb_trades": len(df),
          "trades_par_jour": len(df) / nj if nj else 0.0,
          "capital_final": equity, "rejets": rej}
    if not df.empty:
        st |= {"winrate": (df.pnl > 0).mean(), "R_moyen": df.R.mean(),
               "R_ecart": df.R.std(ddof=1),
               "stop_median": df.stop_dist.median()}
    return df, st
