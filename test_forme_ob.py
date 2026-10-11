"""La forme de la bougie d'Order Block tient-elle comme filtre ?

L'etude des dix attributs a fait ressortir un seul candidat serieux : la
part du corps dans l'amplitude de la bougie d'Order Block. Son interet ne
tient pas a la taille de son effet mais a sa STABILITE. L'ecart entre le
quintile haut et le quintile bas vaut -0.302 sur l'echantillon complet,
-0.303 sur la premiere moitie et -0.304 sur la seconde, avec une
progression monotone dans les trois blocs. Depuis le debut de ce projet,
tout ce qui paraissait prometteur s'inversait entre les deux moities.

CE QUE CE SCRIPT VERIFIE, ET QUE LES QUINTILES NE DISENT PAS
Un classement en quintiles decrit la population des trades deja pris. Un
FILTRE, lui, change le deroulement : refuser une zone libere la place pour
la suivante, decale les comptes journaliers et modifie l'echantillon. Le
filtre est donc applique dans le moteur, a la detection, et non a
posteriori sur le tableau des resultats.

Quatre seuils sont compares a l'absence de filtre, chacun contre les memes
temoins, sur l'echantillon complet puis sur chaque moitie. Un seuil n'est
retenu que si son ecart au temoin reste du meme cote des deux cotes de la
coupure, et s'il ne detruit pas la frequence au passage : un filtre qui ne
laisse qu'un trade par semaine n'est pas exploitable, quel que soit son R.

    python test_forme_ob.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quant.mt5_loader import load_mt5_csv, resample
from quant.scalp_mtf import MTFConfig, backtest_mtf
from quant.surrogate import shuffle_bars

N_TEMOINS = 4
GRAINE = 20261011
SEUILS = [1.0, 0.6, 0.5, 0.4, 0.3]


def mesure(m1: pd.DataFrame, seuil: float) -> dict:
    cfg = MTFConfig(min_trend_force=0.0, partial_r=1e9, max_body_frac=seuil)
    tr, st = backtest_mtf(resample(m1, "5min"), m1, cfg)
    if tr.empty:
        return {}
    return {"n": len(tr), "jour": st["trades_par_jour"],
            "wr": st["winrate"] * 100, "R": tr.R.mean(), "sd": tr.R.std(ddof=1)}


def bloc(m1: pd.DataFrame, titre: str, graine: int) -> dict:
    jours = resample(m1, "5min").index.normalize().nunique()
    print(f"\n{titre}  ({jours} jours)")
    print("  corps max      n   freq    WR     R reel   temoin   ecart     t")
    print("  " + "-" * 66)
    temoins = [shuffle_bars(m1, seed=graine + k) for k in range(N_TEMOINS)]
    out = {}
    for s in SEUILS:
        r = mesure(m1, s)
        if not r or r["n"] < 50:
            print(f"  {s:<10} {r.get('n', 0):>5}  (trop peu de trades)")
            continue
        rs = [mesure(t1, s) for t1 in temoins]
        rs = [x for x in rs if x]
        rt = float(np.mean([x["R"] for x in rs])) if rs else np.nan
        ec = r["R"] - rt
        t = ec / (r["sd"] / np.sqrt(r["n"])) if r["sd"] > 0 else np.nan
        out[s] = (ec, r["jour"])
        nom = "aucun filtre" if s >= 1.0 else f"<= {s:.1f}"
        print(f"  {nom:<10} {r['n']:>5} {r['jour']:>5.2f} {r['wr']:5.1f}%  "
              f"{r['R']:+.3f}  {rt:+.3f}  {ec:+.3f}  {t:+5.2f}")
    return out


def main(chemin: str) -> None:
    m1 = resample(load_mt5_csv(chemin), "1min")
    print(f"{len(m1):,} bougies M1 | filtre applique A LA DETECTION,")
    print("donc le deroulement change : refuser une zone libere la place.")

    tout = bloc(m1, "ECHANTILLON COMPLET", GRAINE)
    coupe = m1.index[len(m1) // 2]
    a = bloc(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100)
    b = bloc(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200)

    print("\n\nSTABILITE")
    print("  corps max    complet   1re      2e     freq    verdict")
    print("  " + "-" * 62)
    base = tout.get(1.0, (np.nan, np.nan))[0]
    for s in SEUILS:
        if s not in tout:
            continue
        nt, fr = tout[s]
        va = a.get(s, (np.nan,))[0]
        vb = b.get(s, (np.nan,))[0]
        if np.isnan(va) or np.isnan(vb):
            v = "indeterminable"
        elif va > 0 and vb > 0:
            v = "POSITIF DES DEUX COTES"
        elif va * vb < 0:
            v = "s'inverse"
        else:
            v = "negatif des deux cotes"
        nom = "aucun" if s >= 1.0 else f"<= {s:.1f}"
        print(f"  {nom:<10} {nt:+.3f}   {va:+.3f}  {vb:+.3f}  {fr:4.2f}/j  {v}")

    print("\nLECTURE")
    print(f"  Reference sans filtre : ecart {base:+.3f} R.")
    print("  Un seuil n'apporte quelque chose que s'il fait MIEUX que cette")
    print("  reference ET garde le meme sens sur les deux moities ET conserve")
    print("  assez de trades pour etre tradable. Les trois conditions, pas une.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
