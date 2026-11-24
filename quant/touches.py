"""Un niveau souvent touche est-il plus solide, ou plus fragile ?

LA CROYANCE
Elle est affichee sur la capture : "SUPPORT - 5 TOUCHES", "RESISTANCE -
4 TOUCHES". Compter les touches sous-entend que cela renforce le niveau, et
qu'un support touche cinq fois est plus sur qu'un support touche une fois.

LE RAISONNEMENT DE MARCHE DIT L'INVERSE
Un support ne tient que tant qu'il reste des ordres d'achat poses dessous.
Chaque test en consomme une partie. Un niveau touche cinq fois a donc deja
ete vide quatre fois, et devrait casser PLUS facilement qu'un niveau neuf.
Les deux affirmations ne peuvent pas etre vraies ensemble, et la mesure
tranche.

COMMENT C'EST MESURE
Les niveaux sont les sommets et creux confirmes en M5, lus avec leur retard
de confirmation. Chaque niveau est ensuite suivi minute par minute :

  TOUCHE    le prix revient a moins de `tolerance` du niveau
  REBOND    il s'en eloigne de `seuil_rebond` avant de le traverser
  CASSURE   il le traverse de `seuil_cassure` avant de s'en eloigner

Apres un rebond, le compteur du niveau augmente et l'on continue. Apres une
cassure, le niveau est mort. On obtient ainsi, pour chaque rang de touche,
un taux de rebond : c'est la seule chose a regarder.

CE QUI REND LE TEST CONCLUANT
Les deux seuils forment deux barrieres, donc sur une marche aleatoire le
taux de rebond est connu d'avance, et surtout il ne depend PAS du rang de
la touche. Le moteur est ainsi verifiable en niveau comme en pente, ce qui
manquait au moteur d'Order Block, lequel anticipait sans que rien ne le
signale.

ATTENTION AU NIVEAU DE REFERENCE. Une touche est declaree des que le prix
passe a `tolerance` du niveau, et non quand il l'atteint : au moment de la
touche il est donc encore eloigne de la barriere de cassure d'a peu pres
`tolerance`. Le taux attendu vaut par consequent

    (seuil_cassure + tolerance) / (seuil_cassure + seuil_rebond)

soit 44.4% avec les valeurs par defaut, et non les 33.3% que donnerait une
touche exactement sur le niveau. Mesure sur trois marches aleatoires :
44.1%, 43.7% et 44.2%. L'accord est net, le moteur est juste.

ATTENTION A L'ERREUR-TYPE. Les touches d'un meme niveau ne sont pas
independantes : elles viennent du meme chemin de prix. L'erreur-type que
renvoie la regression vaut 0.29 point par touche, alors que la dispersion
entre trois marches aleatoires vaut environ 1.0. Elle est donc sous-estimee
d'un facteur trois a quatre, et les t calcules a l'interieur d'un seul
echantillon n'ont aucun sens : l'un des trois tirages aleatoires affiche un
t de +4.79 alors que la verite est nulle par construction.
La pente ne doit donc etre jugee que contre une distribution de temoins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TouchConfig:
    fractal_k: int = 3            # demi-largeur du pivot, en bougies M5
    atr_bars: int = 288           # 24 heures de bougies M5
    tolerance: float = 0.5        # proximite comptant comme une touche, en ATR
    seuil_cassure: float = 1.5    # traversee comptant comme cassure, en ATR
    seuil_rebond: float = 3.0     # eloignement comptant comme rebond, en ATR
    duree_vie_min: int = 2880     # duree de vie d'un niveau, en minutes
    rang_max: int = 8             # au-dela, les rangs sont regroupes


def _atr(df: pd.DataFrame, w: int) -> np.ndarray:
    prev = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(),
                    (df.low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(w, min_periods=w // 2).mean().to_numpy()


def _pivots(h: np.ndarray, l: np.ndarray, k: int):
    n = len(h)
    sh = np.zeros(n, bool)
    sl = np.zeros(n, bool)
    for i in range(k, n - k):
        f = slice(i - k, i + k + 1)
        if h[i] == h[f].max() and (h[f] == h[i]).sum() == 1:
            sh[i] = True
        if l[i] == l[f].min() and (l[f] == l[i]).sum() == 1:
            sl[i] = True
    return sh, sl


def evenements(m5: pd.DataFrame, m1: pd.DataFrame,
               cfg: TouchConfig) -> pd.DataFrame:
    """Un enregistrement par TOUCHE, avec son rang et son issue."""
    h5 = m5.high.to_numpy()
    l5 = m5.low.to_numpy()
    atr5 = _atr(m5, cfg.atr_bars)
    sh, sl = _pivots(h5, l5, cfg.fractal_k)

    hh = m1.high.to_numpy()
    ll = m1.low.to_numpy()
    pos1 = m1.index.values.astype("datetime64[ns]")
    n1 = len(hh)

    out = []
    for idx in np.flatnonzero(sh | sl):
        # Le pivot n'est CONNU qu'apres ses k bougies de confirmation.
        conf = idx + cfg.fractal_k
        if conf >= len(h5) or np.isnan(atr5[conf]) or atr5[conf] <= 0:
            continue
        a = int(np.searchsorted(pos1, m5.index[conf].to_datetime64(),
                                side="right"))
        b = min(a + cfg.duree_vie_min, n1)
        if b - a < 60:
            continue

        for support in ([True] if sl[idx] else []) + \
                       ([False] if sh[idx] else []):
            niveau = l5[idx] if support else h5[idx]
            atr = atr5[conf]
            tol = cfg.tolerance * atr
            cass = cfg.seuil_cassure * atr
            reb = cfg.seuil_rebond * atr

            seg_lo = ll[a:b]
            seg_hi = hh[a:b]
            m = len(seg_lo)

            # Le prix doit d'abord S'ELOIGNER du niveau, sans quoi la bougie
            # qui a forme le pivot compterait elle-meme comme une touche.
            loin = (seg_lo > niveau + tol) if support \
                else (seg_hi < niveau - tol)
            if not loin.any():
                continue
            p = int(np.argmax(loin))

            rang = 0
            while p < m and rang < 50:
                proche = (seg_lo[p:] <= niveau + tol) if support \
                    else (seg_hi[p:] >= niveau - tol)
                if not proche.any():
                    break
                t = p + int(np.argmax(proche))

                perce = (seg_lo[t:] <= niveau - cass) if support \
                    else (seg_hi[t:] >= niveau + cass)
                part = (seg_hi[t:] >= niveau + reb) if support \
                    else (seg_lo[t:] <= niveau - reb)
                i_c = int(np.argmax(perce)) if perce.any() else m + 1
                i_r = int(np.argmax(part)) if part.any() else m + 1
                if i_c == m + 1 and i_r == m + 1:
                    break
                rang += 1
                # Egalite a l'interieur d'une bougie : on tranche toujours
                # dans le meme sens, le biais s'annulant d'un rang a l'autre.
                rebond = i_r < i_c
                out.append({"ts": m1.index[a + t], "support": support,
                            "rang": min(rang, cfg.rang_max),
                            "rebond": int(rebond), "atr": atr})
                if not rebond:
                    break
                p = t + i_r + 1

    return pd.DataFrame(out)


def taux_par_rang(ev: pd.DataFrame, rang_max: int = 8) -> pd.DataFrame:
    """Taux de rebond par rang de touche, avec son erreur-type."""
    if ev.empty:
        return pd.DataFrame()
    g = ev.groupby("rang").rebond.agg(["count", "mean"])
    g["se"] = np.sqrt(g["mean"] * (1 - g["mean"]) / g["count"])
    g.columns = ["n", "taux", "se"]
    return g


def pente(ev: pd.DataFrame) -> tuple[float, float, float]:
    """Pente du taux de rebond en fonction du rang, par regression simple.

    Renvoie la pente en points de pourcentage par touche, son erreur-type et
    le t correspondant. Une pente NEGATIVE veut dire qu'un niveau souvent
    touche casse plus facilement, donc que la croyance est fausse.
    """
    if ev.empty or ev.rang.nunique() < 3:
        return np.nan, np.nan, np.nan
    x = ev.rang.to_numpy(float)
    y = ev.rebond.to_numpy(float)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx <= 0:
        return np.nan, np.nan, np.nan
    b = ((x - xm) * (y - y.mean())).sum() / sxx
    resid = y - (y.mean() + b * (x - xm))
    s2 = (resid ** 2).sum() / (len(x) - 2)
    se = np.sqrt(s2 / sxx)
    return b * 100, se * 100, (b / se if se > 0 else np.nan)
