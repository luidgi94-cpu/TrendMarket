"""Le resultat est-il un effet, ou une case chanceuse ?

LA QUESTION, POSEE CORRECTEMENT
Une configuration de cassure-retest tient les deux conditions a la fois :
ecart au temoin positif sur les deux moities, et R absolu positif sur les
deux moities. C'est la bande a deux ecarts-types avec un stop de deux
ecarts-types. Mais une soixantaine de configurations ont deja ete essayees
dans ce projet. A ce stade, qu'une case ressorte ne prouve rien : avec
assez d'essais, il en ressort toujours une.

Ce qui distingue un effet d'une case chanceuse n'est pas son t, c'est la
FORME DE LA SURFACE autour d'elle. Un effet reel varie regulierement quand
on bouge les reglages, et son voisinage entier penche du meme cote. Une
case chanceuse est entouree de cases quelconques.

Deux axes sont balayes, et ils ont chacun une raison d'etre.

  k, la largeur de la bande, commande la FREQUENCE. L'utilisateur accepte
  deux trades par jour, ce que la configuration retenue ne donne pas : elle
  en produit 1.31. Une bande plus etroite declenche plus souvent.

  stop_sigma commande le COUT. C'est lui qui a fait basculer le signe : a
  stop serre les frais valent 6.3% du risque, a stop large 1.6%. L'ecart au
  temoin, lui, bougeait peu. Il faut verifier que cette relation est
  monotone et non un accident.

CE QUI SERA CONCLUANT, ET CE QUI NE LE SERA PAS
Concluant : une plage CONTINUE de cases positives des deux cotes de la
coupure, avec une progression reguliere selon stop_sigma. Non concluant :
des cases positives dispersees, meme nombreuses, ou un maximum colle au
bord de la grille, qui signalerait que l'optimum est dehors.

    python test_retest_carte.py chemin/vers/XAUUSD_M1.csv
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

KS = [1.0, 1.5, 2.0, 2.5]
STOPS = [1.0, 1.5, 2.0, 3.0]


def mesure(m1: pd.DataFrame, k: float, ss: float) -> dict:
    tr, st = backtest_retest(m1, RetestConfig(
        k=k, stop_sigma=ss, retrace_sigma=0.5, cible="r", cible_r=2.0))
    if tr.empty:
        return {}
    return {"n": len(tr), "jour": st["trades_par_jour"],
            "wr": st["winrate"] * 100, "R": tr.R.mean(),
            "sd": tr.R.std(ddof=1), "stop": tr.stop_dist.median()}


def bloc(m1: pd.DataFrame, titre: str, graine: int, rng_m1: float) -> dict:
    jours = m1.index.normalize().nunique()
    print(f"\n{titre}  ({jours} jours)", flush=True)
    print("     k  stop      n   freq    WR  stop$ st/M1     R  EUR/mois"
          "  temoin   ecart     t", flush=True)
    print("  " + "-" * 86, flush=True)
    tem = [shuffle_bars(m1, seed=graine + i) for i in range(N_TEMOINS)]
    out = {}
    for k in KS:
        for ss in STOPS:
            r = mesure(m1, k, ss)
            if not r or r["n"] < 30:
                print(f"  {k:>4.1f}{ss:>6.1f}  {r.get('n', 0):>5}"
                      f"  (trop peu)", flush=True)
                continue
            rs = [x for x in (mesure(t, k, ss) for t in tem) if x]
            rt = float(np.mean([x["R"] for x in rs])) if rs else np.nan
            ec = r["R"] - rt
            t = ec / (r["sd"] / np.sqrt(r["n"])) if r["sd"] > 0 else np.nan
            eur = r["R"] * RISQUE_EUR * r["jour"] * JOURS_MOIS
            ratio = r["stop"] / rng_m1
            out[(k, ss)] = (ec, r["jour"], r["R"], ratio)
            print(f"  {k:>4.1f}{ss:>6.1f}  {r['n']:>5} {r['jour']:>5.2f} "
                  f"{r['wr']:5.1f}% {r['stop']:6.2f} {ratio:5.1f}"
                  f"{' ' if ratio >= 5 else '!'}{r['R']:+.3f} {eur:>+8.0f}"
                  f"  {rt:+.3f}  {ec:+.3f}  {t:+5.2f}", flush=True)
    return out


def carte(titre: str, d: dict, idx: int, fmt: str = "{:+9.3f}") -> None:
    print(f"\n  {titre}")
    print("  k \\ stop" + "".join(f"{s:>9.1f}" for s in STOPS))
    for k in KS:
        ligne = f"  {k:>8.1f}"
        for s in STOPS:
            v = d.get((k, s))
            ligne += "      --- " if v is None else fmt.format(v[idx])
        print(ligne)


def main(chemin: str) -> None:
    t0 = time.time()
    m1 = charge(chemin)
    rng_m1 = float((m1.high - m1.low).median())
    print(f"{len(m1):,} bougies M1 | retour 0.5 ecart-type, cible 2 R")
    print(f"amplitude mediane d'une bougie M1 : {rng_m1:.3f} USD")

    tout = bloc(m1, "ECHANTILLON COMPLET", GRAINE, rng_m1)
    coupe = m1.index[len(m1) // 2]
    a = bloc(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100, rng_m1)
    b = bloc(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200, rng_m1)

    print("\n\nCARTES")
    carte("ecart au temoin, echantillon complet", tout, 0)
    carte("ecart au temoin, premiere moitie", a, 0)
    carte("ecart au temoin, seconde moitie", b, 0)
    carte("R absolu, echantillon complet", tout, 2)
    carte("trades par jour", tout, 1, "{:>9.2f}")

    print("\n\nCASES QUI TIENNENT TOUT  (ecart ET R positifs des deux cotes)")
    bonnes = []
    for cle, v in tout.items():
        va, vb = a.get(cle), b.get(cle)
        if va is None or vb is None:
            continue
        if va[0] > 0 and vb[0] > 0 and va[2] > 0 and vb[2] > 0:
            bonnes.append((cle, v[0], v[1], v[2], v[3]))
    if not bonnes:
        print("  aucune.")
    for (k, ss), ec, fr, R, ratio in sorted(bonnes, key=lambda x: -x[1]):
        print(f"    k={k:.1f} stop={ss:.1f}  ecart {ec:+.3f}  R {R:+.3f}  "
              f"{fr:.2f}/jour  stop {ratio:.1f} bougies M1")
    print(f"\n  {len(bonnes)} case(s) sur {len(tout)}. Pour conclure il faut")
    print("  qu'elles se TOUCHENT sur la carte et que la progression selon")
    print("  stop_sigma soit reguliere. Des cases dispersees restent du")
    print("  hasard, meme nombreuses.")
    print(f"\n  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
