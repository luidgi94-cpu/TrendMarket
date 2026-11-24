"""Cassure de bande puis retest, a l'ordre limite : la synthese du projet.

POURQUOI CETTE FORME PLUTOT QU'UNE AUTRE
Elle reunit les trois seules choses que ce projet ait mesurees solidement.

  L'EVENEMENT. La bande de VWAP porte le plus gros effet de tout le projet :
  la jouer en retour a la moyenne perd 0.165 R de PLUS que le hasard, avec
  un t de -4.3, sur les deux moities et pour toutes les largeurs essayees.
  Aucune autre mesure n'a ete aussi grosse ni aussi stable.

  LE SENS. Si fader perd autant, le sens qui porte l'information est
  l'autre : aller avec le mouvement donne +0.104 R face au temoin, positif
  sur les deux moities.

  L'EXECUTION. Mais entrer a la cassure coute 0.198 R avant toute lecture du
  marche, un ordre stop remplissant APRES le depart du mouvement. L'ordre
  limite, lui, a un temoin a peu pres nul. D'ou l'entree sur le RETOUR vers
  la bande, et non sur la cassure.

LE MOTEUR A ETE VERIFIE, ET IL A FALLU LE CORRIGER
Premiere version : +0.226 R sur marche aleatoire, ou la reponse doit etre
zero. Le defaut s'est trahi par sa signature, le biais grandissant avec la
duree de vie de l'ordre : +0.049 a quinze minutes, +0.286 a quatre cent
quatre-vingts. Le moteur reposait un ordre a chaque bougie tant que le prix
restait au-dela de la bande et gardait le premier rempli, si bien que le
choix de l'ordre retenu dependait du futur. Un evenement ne donne desormais
qu'un seul ordre. Apres correction : -0.045 a +0.005 selon les reglages,
tous avec un t inferieur a 1.4, et un taux de reussite de 31.9 a 40.0%
contre 33.3% theorique.

    python test_retest.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from quant.vwap import RetestConfig, backtest_retest
from quant.surrogate import shuffle_bars
from test_momentum import charge

N_TEMOINS = 4
GRAINE = 20261124
RISQUE_EUR = 4.50
JOURS_MOIS = 21

CONFIGS = [
    ("k2.0 ret0.5 stop1.0", dict(k=2.0, retrace_sigma=0.5, stop_sigma=1.0)),
    ("k2.0 ret0.3 stop1.0", dict(k=2.0, retrace_sigma=0.3, stop_sigma=1.0)),
    ("k2.0 ret0.8 stop1.0", dict(k=2.0, retrace_sigma=0.8, stop_sigma=1.0)),
    ("k1.5 ret0.5 stop1.0", dict(k=1.5, retrace_sigma=0.5, stop_sigma=1.0)),
    ("k2.5 ret0.5 stop1.0", dict(k=2.5, retrace_sigma=0.5, stop_sigma=1.0)),
    ("k2.0 ret0.5 stop0.5", dict(k=2.0, retrace_sigma=0.5, stop_sigma=0.5)),
    ("k2.0 ret0.5 stop2.0", dict(k=2.0, retrace_sigma=0.5, stop_sigma=2.0)),
    ("k2.0 ret0.5 cible1R", dict(k=2.0, retrace_sigma=0.5, stop_sigma=1.0,
                                 cible_r=1.0)),
    ("k2.0 ret0.5 cible3R", dict(k=2.0, retrace_sigma=0.5, stop_sigma=1.0,
                                 cible_r=3.0)),
    ("k2.0 poids plat",     dict(k=2.0, retrace_sigma=0.5, stop_sigma=1.0,
                                 poids="plat")),
]


def mesure(m1: pd.DataFrame, opts: dict) -> dict:
    tr, st = backtest_retest(m1, RetestConfig(cible="r", **opts))
    if tr.empty:
        return {}
    return {"n": len(tr), "jour": st["trades_par_jour"],
            "wr": st["winrate"] * 100, "R": tr.R.mean(),
            "sd": tr.R.std(ddof=1), "stop": tr.stop_dist.median(),
            "cr": tr.cible_r.median()}


def bloc(m1: pd.DataFrame, titre: str, graine: int, rng_m1: float) -> dict:
    jours = m1.index.normalize().nunique()
    print(f"\n{titre}  ({jours} jours)", flush=True)
    print("  configuration           n   freq    WR  stop stop/M1 cible"
          "     R   EUR/mois  temoin   ecart     t")
    print("  " + "-" * 103, flush=True)
    tem = [shuffle_bars(m1, seed=graine + k) for k in range(N_TEMOINS)]
    out = {}
    for nom, opts in CONFIGS:
        r = mesure(m1, opts)
        if not r or r["n"] < 30:
            print(f"  {nom:<21} {r.get('n', 0):>5}  (trop peu de trades)",
                  flush=True)
            continue
        rs = [x for x in (mesure(t, opts) for t in tem) if x]
        rt = float(np.mean([x["R"] for x in rs])) if rs else np.nan
        ec = r["R"] - rt
        t = ec / (r["sd"] / np.sqrt(r["n"])) if r["sd"] > 0 else np.nan
        eur = r["R"] * RISQUE_EUR * r["jour"] * JOURS_MOIS
        ratio = r["stop"] / rng_m1
        out[nom] = (ec, r["jour"], r["R"], r["wr"], eur, ratio)
        print(f"  {nom:<21} {r['n']:>5} {r['jour']:>5.2f} {r['wr']:5.1f}% "
              f"{r['stop']:5.2f} {ratio:5.1f}{' ' if ratio >= 5 else '!'} "
              f"{r['cr']:4.2f}R {r['R']:+.3f} {eur:>+8.0f}  {rt:+.3f}  "
              f"{ec:+.3f}  {t:+5.2f}", flush=True)
    return out


def main(chemin: str) -> None:
    t0 = time.time()
    m1 = charge(chemin)
    rng_m1 = float((m1.high - m1.low).median())
    print(f"{len(m1):,} bougies M1 | "
          f"{m1.index[0]:%d/%m/%Y} au {m1.index[-1]:%d/%m/%Y}")
    print(f"amplitude mediane d'une bougie M1 : {rng_m1:.3f} USD")
    print("MOTEUR VERIFIE : rend -0.045 a +0.005 R sur marche aleatoire")
    print("sans derive, ou la reponse doit etre zero.")

    tout = bloc(m1, "ECHANTILLON COMPLET", GRAINE, rng_m1)
    coupe = m1.index[len(m1) // 2]
    a = bloc(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100, rng_m1)
    b = bloc(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200, rng_m1)

    print("\n\nVERDICT")
    print("  configuration            WR      R   EUR/mois  ecart"
          "      1re      2e     verdict")
    print("  " + "-" * 95)
    for nom, _ in CONFIGS:
        if nom not in tout:
            continue
        ec, fr, R, wr, eur, ratio = tout[nom]
        va = a.get(nom, (np.nan,))[0]
        vb = b.get(nom, (np.nan,))[0]
        if np.isnan(va) or np.isnan(vb):
            v = "indeterminable"
        elif va > 0 and vb > 0:
            v = "POSITIF DES DEUX COTES"
        elif va * vb < 0:
            v = "s'inverse"
        else:
            v = "negatif des deux cotes"
        print(f"  {nom:<21} {wr:5.1f}% {R:+.3f} {eur:>+8.0f}  {ec:+.3f}"
              f"   {va:+.3f}  {vb:+.3f}   {v}")

    print("\n  Deux choses doivent tenir ensemble, et aucune ne suffit seule :")
    print("  l'ecart au temoin positif sur les DEUX moities, et un R absolu")
    print("  positif. Un ecart positif avec un R negatif signifie seulement")
    print("  que le pari est moins mauvais que le hasard.")
    print(f"\n  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
