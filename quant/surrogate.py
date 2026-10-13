"""Construction d'un marche temoin a partir de donnees REELLES.

Sur donnees synthetiques, le monde de controle etait fabrique. Sur donnees
reelles il faut le deriver du marche lui-meme, sinon un resultat positif
reste ininterpretable : impossible de distinguer un edge d'un artefact de
geometrie.

Methode : on melange les bougies M1 A L'INTERIEUR de chaque heure de la
journee. Sont conserves le profil de volatilite horaire, la distribution
des rendements, les queues epaisses et la forme des bougies. Est detruite
la seule chose qui puisse porter un edge : l'ordre des evenements, donc
toute structure, tout momentum, tout Order Block.

Une strategie qui gagne autant sur ce temoin que sur le marche reel ne
capture aucune information : elle exploite sa propre geometrie.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def shuffle_bars(m1: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Melange les bougies M1 par heure de la journee, puis reconstruit le prix.

    Chaque bougie est traitee comme un bloc indissociable : son rendement et
    sa forme (positions du haut et du bas relativement a la cloture) voyagent
    ensemble. Le chemin de prix est ensuite reconstruit par composition.
    """
    rng = np.random.default_rng(seed)
    o = m1.open.to_numpy(); h = m1.high.to_numpy()
    l = m1.low.to_numpy(); c = m1.close.to_numpy()

    ret = np.zeros(len(c))
    ret[1:] = np.log(c[1:] / c[:-1])
    # forme de la bougie, en proportion de sa propre cloture
    up = np.log(np.maximum(h, c) / c)
    dn = np.log(np.minimum(l, c) / c)
    op = np.log(o / c)

    order = np.arange(len(c))
    hours = m1.index.hour.to_numpy()
    for hh in np.unique(hours):
        m = hours == hh
        idx = order[m]
        order[m] = rng.permutation(idx)

    ret_s, up_s, dn_s, op_s = ret[order], up[order], dn[order], op[order]
    logc = np.log(c[0]) + np.cumsum(ret_s)
    cs = np.exp(logc)

    out = pd.DataFrame({
        "open":  cs * np.exp(op_s),
        "high":  cs * np.exp(up_s),
        "low":   cs * np.exp(dn_s),
        "close": cs,
    }, index=m1.index)
    # coherence OHLC apres recomposition
    out["high"] = out[["open", "high", "close"]].max(axis=1)
    out["low"] = out[["open", "low", "close"]].min(axis=1)
    for extra in ("tickvol", "spread_usd"):
        if extra in m1.columns:
            out[extra] = m1[extra].to_numpy()[order]
    return out


def shuffle_blocks(m1: pd.DataFrame, block_minutes: int,
                   seed: int = 0) -> pd.DataFrame:
    """Melange des BLOCS de bougies M1, en conservant l'interieur de chaque bloc.

    POURQUOI CE SECOND TEMOIN EXISTE
    shuffle_bars melange les bougies M1 une par une. Pour une detection en M5
    cela va, parce qu'une bougie M5 ne contient que cinq bougies M1 et que sa
    forme change donc peu. Pour une detection en H1, une bougie du temoin est
    reconstituee a partir de soixante bougies M1 tirees de soixante journees
    differentes : leurs rendements etant devenus independants, l'amplitude de
    la bougie croit comme la racine de soixante au lieu de suivre le regime
    reel. Les bougies H1 du temoin ne ressemblent alors plus a des bougies H1,
    les zones n'ont plus la meme epaisseur, les stops plus la meme taille, et
    la comparaison ne porte plus sur la meme geometrie.

    Ce temoin-ci melange des blocs de la duree de l'unite de detection. A
    l'interieur d'un bloc, le chemin reel est conserve INTACT : les bougies de
    detection du temoin sont donc de vraies bougies, avec leur vraie forme et
    leur vraie amplitude. Seul leur ORDRE est detruit, c'est-a-dire exactement
    ce qu'un Order Block pretend exploiter. C'est l'hypothese nulle juste :
    les formes sont vraies, la suite est le hasard.

    Les blocs ne sont permutes qu'entre blocs de MEME HEURE de la journee, ce
    qui preserve le profil de volatilite horaire, et de MEME TAILLE, ce qui
    garantit qu'un bloc deplace retombe exactement sur un emplacement de meme
    duree. Sans cette seconde condition, un bloc incomplet de fin de semaine
    decalerait toutes les bougies suivantes.
    """
    rng = np.random.default_rng(seed)
    o = m1.open.to_numpy(); h = m1.high.to_numpy()
    l = m1.low.to_numpy(); c = m1.close.to_numpy()

    ret = np.zeros(len(c))
    ret[1:] = np.log(c[1:] / c[:-1])
    up = np.log(np.maximum(h, c) / c)
    dn = np.log(np.minimum(l, c) / c)
    op = np.log(o / c)

    # Identifiant de bloc : horodatage tronque a la duree du bloc. L'index
    # etant croissant, les codes le sont aussi, ce qui permet de delimiter
    # chaque bloc par une simple recherche dichotomique.
    codes, uniq = pd.factorize(m1.index.floor(f"{block_minutes}min"),
                               sort=True)
    nb = len(uniq)
    debuts = np.searchsorted(codes, np.arange(nb), side="left")
    fins = np.searchsorted(codes, np.arange(nb), side="right")
    tailles = fins - debuts
    heures = pd.DatetimeIndex(uniq).hour.to_numpy()

    nouveau = np.arange(nb)
    for cle in {(hh, t) for hh, t in zip(heures, tailles)}:
        m = (heures == cle[0]) & (tailles == cle[1])
        nouveau[m] = rng.permutation(nouveau[m])

    order = np.concatenate([np.arange(debuts[b], fins[b]) for b in nouveau])
    assert len(order) == len(c)

    ret_s, up_s, dn_s, op_s = ret[order], up[order], dn[order], op[order]
    cs = np.exp(np.log(c[0]) + np.cumsum(ret_s))
    out = pd.DataFrame({
        "open":  cs * np.exp(op_s),
        "high":  cs * np.exp(up_s),
        "low":   cs * np.exp(dn_s),
        "close": cs,
    }, index=m1.index)
    out["high"] = out[["open", "high", "close"]].max(axis=1)
    out["low"] = out[["open", "low", "close"]].min(axis=1)
    for extra in ("tickvol", "spread_usd"):
        if extra in m1.columns:
            out[extra] = m1[extra].to_numpy()[order]
    return out


def resample_m5(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "tickvol" in df.columns: agg["tickvol"] = "sum"
    if "spread_usd" in df.columns: agg["spread_usd"] = "mean"
    return df.resample("5min").agg(agg).dropna(subset=["open", "close"])
