"""L'avantage apparent de H1 survit-il a un temoin correct ?

CE QUI A ETE TROUVE, ET POURQUOI IL FAUT S'EN MEFIER
La detection en H1, sans aucun filtre, est la premiere configuration de tout
le projet a afficher un ecart positif au temoin sur l'echantillon complet
ET sur les deux moities : +0.163 R avec un t de 4.02, puis +0.167 et +0.101
avec des t de 2.30 et 2.63. Aucune des neuf familles d'hypotheses testees
auparavant n'avait tenu ce test.

Mais le temoin utilise melange les bougies M1 UNE PAR UNE. Une bougie H1 du
temoin est alors reconstituee a partir de soixante bougies M1 tirees de
soixante journees differentes. Les rendements M1 de l'or etant legerement
anti-correles, les melanger GONFLE l'amplitude horaire : 16.48 USD d'ecart
median contre 12.67 en reel, soit trente pour cent de trop. Les zones du
temoin sont donc plus epaisses, ses stops plus gros, et la cible fixe de
100 pips y vaut moins de R. Le temoin est handicape par sa geometrie, et
l'ecart mesure pourrait n'etre que ce handicap.

LE TEMOIN JUSTE
shuffle_blocks melange des BLOCS de la duree de l'unite de detection, en
conservant intact le chemin a l'interieur de chaque bloc. Les bougies de
detection du temoin sont donc de vraies bougies, avec leur vraie amplitude :
11.12 USD contre 12.67 en reel, l'ecart restant venant de ce que les blocs
retombent a des niveaux de prix differents. Seul l'ORDRE des bougies est
detruit, c'est-a-dire exactement ce qu'un Order Block pretend exploiter.

Les deux temoins sont affiches cote a cote, pour que la difference entre
les deux soit visible et non pas seulement affirmee.

    python test_unites_temoin_juste.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from quant.mt5_loader import load_mt5_csv, resample
from quant.scalp_mtf import MTFConfig, backtest_mtf
from quant.surrogate import shuffle_bars, shuffle_blocks

N_TEMOINS = 4
GRAINE = 20261013
UNITES = [("M5", "5min", 5), ("M15", "15min", 15), ("M30", "30min", 30),
          ("H1", "1h", 60), ("H4", "4h", 240)]


def config(tf_min: int) -> MTFConfig:
    return MTFConfig(min_trend_force=0.0, partial_r=1e9, max_trades_per_day=50,
                     entry_mode="contact", tp_usd=10.0,
                     tf_minutes=tf_min,
                     atr_bars=max(14, 1440 // tf_min),
                     max_hold_min=48 * tf_min)


def mesure(m1: pd.DataFrame, regle: str, tf_min: int) -> dict:
    tr, st = backtest_mtf(resample(m1, regle), m1, config(tf_min))
    if tr.empty:
        return {}
    return {"n": len(tr), "wr": st["winrate"] * 100, "R": tr.R.mean(),
            "sd": tr.R.std(ddof=1), "stop": tr.stop_dist.median()}


def bloc(m1: pd.DataFrame, titre: str, graine: int) -> dict:
    jours = m1.index.normalize().nunique()
    print(f"\n{titre}  ({jours} jours)", flush=True)
    print("  unite     n    WR    stop     R reel  | temoin BARS  stop"
          "  ecart    t  | temoin BLOCS  stop  ecart    t")
    print("  " + "-" * 104, flush=True)
    bars = [shuffle_bars(m1, seed=graine + k) for k in range(N_TEMOINS)]
    out = {}
    for nom_u, regle, tf_min in UNITES:
        r = mesure(m1, regle, tf_min)
        if not r or r["n"] < 30:
            print(f"  {nom_u:<6} {r.get('n', 0):>5}  (trop peu de trades)",
                  flush=True)
            continue
        se = r["sd"] / np.sqrt(r["n"])
        ligne = f"  {nom_u:<6} {r['n']:>5} {r['wr']:5.1f}% {r['stop']:6.2f} " \
                f"{r['R']:+.3f}  |"
        res = {}
        for cle, jeu in (("bars", bars),
                         ("blocs", [shuffle_blocks(m1, tf_min, seed=graine + k)
                                    for k in range(N_TEMOINS)])):
            rs = [x for x in (mesure(t, regle, tf_min) for t in jeu) if x]
            if not rs:
                res[cle] = (np.nan, np.nan, np.nan)
                ligne += "      --            |"
                continue
            rt = float(np.mean([x["R"] for x in rs]))
            stop_t = float(np.mean([x["stop"] for x in rs]))
            ec = r["R"] - rt
            res[cle] = (ec, ec / se if se > 0 else np.nan, stop_t)
            ligne += f" {rt:+.3f} {stop_t:6.2f} {ec:+.3f} {ec / se:+5.2f} |"
        out[nom_u] = res
        print(ligne, flush=True)
    return out


def main(chemin: str) -> None:
    t0 = time.time()
    m1 = resample(load_mt5_csv(chemin), "1min")
    print(f"{len(m1):,} bougies M1 | sans filtre, entree au contact, "
          f"cible 100 pips")
    print("Comparer la colonne 'stop' du reel a celle de chaque temoin : si")
    print("elles divergent, les deux systemes ne jouent pas la meme geometrie")
    print("et l'ecart mesure ne veut rien dire.")

    tout = bloc(m1, "ECHANTILLON COMPLET", GRAINE)
    coupe = m1.index[len(m1) // 2]
    a = bloc(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100)
    b = bloc(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200)

    print("\n\nVERDICT, AVEC LE TEMOIN JUSTE")
    print("  unite   ecart complet    1re      2e     verdict")
    print("  " + "-" * 66)
    for nom_u, _, _ in UNITES:
        if nom_u not in tout:
            continue
        nt = tout[nom_u]["blocs"][0]
        va = a.get(nom_u, {}).get("blocs", (np.nan,))[0]
        vb = b.get(nom_u, {}).get("blocs", (np.nan,))[0]
        if np.isnan(va) or np.isnan(vb):
            v = "indeterminable"
        elif va > 0 and vb > 0:
            v = "POSITIF DES DEUX COTES"
        elif va * vb < 0:
            v = "s'inverse"
        else:
            v = "negatif des deux cotes"
        print(f"  {nom_u:<6} {nt:+.3f}        {va:+.3f}  {vb:+.3f}   {v}")
    print(f"\n  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
