"""Quelle confirmation d'entree vaut quelque chose ?

Le systeme accepte trois signaux d'entree, et il suffit qu'un seul
apparaisse pour prendre position :

    meche_rejet  une meche au moins 1.5 fois plus longue que le corps,
                 du bon cote, avec une cloture dans le bon sens
    englobante   la bougie englobe entierement la precedente, de couleur
                 opposee
    choch        la cloture depasse le dernier extreme de structure
                 oppose, lu avec son decalage de confirmation

Le moteur enregistre depuis le premier jour laquelle a declenche chaque
trade, et cette colonne n'a jamais ete exploitee. C'est pourtant la
question que pose naturellement un utilisateur de l'indicateur : parmi
les trois, laquelle merite qu'on lui fasse confiance.

LE CRITERE, INCHANGE
Un type de confirmation qui affiche un R eleve n'apprend rien s'il
apparait surtout dans des moments volatils, ou le hasard gagne autant.
Chaque type est donc confronte au MEME type mesure sur des temoins, ou
les bougies ont ete melangees a l'interieur de chaque heure. Seul l'ecart
compte.

DEUX PRECAUTIONS
D'abord l'ordre de priorite. La fonction teste la meche, puis
l'englobante, puis le CHoCH, et s'arrete a la premiere trouvee. Les
categories ne sont donc pas symetriques : un trade classe "choch" est un
trade ou ni meche ni englobante n'etaient presentes. On ne compare pas
trois signaux independants mais trois groupes construits par elimination.

Ensuite le nombre de tests. Trois categories examinees apres une
quarantaine de configurations deja essayees sur les memes donnees : un
resultat isole a t = 2 ne vaut rien. Le decoupage temporel est donc
applique d'emblee, et non en rattrapage.

    python test_confirmations.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quant.mt5_loader import load_mt5_csv, resample
from quant.scalp_mtf import MTFConfig, backtest_mtf
from quant.surrogate import shuffle_bars

N_TEMOINS = 4
GRAINE = 20260917
TYPES = ["meche_rejet", "englobante", "choch"]


def par_type(m1: pd.DataFrame) -> dict:
    cfg = MTFConfig(min_trend_force=0.0, partial_r=1e9)
    tr, _ = backtest_mtf(resample(m1, "5min"), m1, cfg)
    if tr.empty:
        return {}
    out = {k: tr.R[tr.confirmation == k] for k in TYPES}
    out["ensemble"] = tr.R
    return out


def bloc(m1: pd.DataFrame, titre: str, graine: int) -> dict:
    jours = resample(m1, "5min").index.normalize().nunique()
    print(f"\n{titre}  ({jours} jours)")
    reel = par_type(m1)
    if not reel:
        print("  aucun trade")
        return {}
    tem = {k: [] for k in reel}
    for k in range(N_TEMOINS):
        g = par_type(shuffle_bars(m1, seed=graine + k))
        for nom, v in g.items():
            if len(v):
                tem[nom].append(v.mean())

    print("  type            n    part    WR      R reel   temoin   ecart     t")
    print("  " + "-" * 68)
    ecarts = {}
    total = len(reel["ensemble"])
    for nom in TYPES + ["ensemble"]:
        v = reel.get(nom, pd.Series(dtype=float))
        if len(v) < 30:
            print(f"  {nom:<13} {len(v):>4}   (trop peu de trades)")
            continue
        rt = float(np.mean(tem[nom])) if tem[nom] else np.nan
        ec = v.mean() - rt
        t = ec / (v.std(ddof=1) / np.sqrt(len(v))) if v.std(ddof=1) > 0 else np.nan
        ecarts[nom] = ec
        print(f"  {nom:<13} {len(v):>4}  {len(v)/total*100:4.1f}%  "
              f"{(v > 0).mean()*100:4.1f}%  {v.mean():+.3f}  {rt:+.3f}  "
              f"{ec:+.3f}  {t:+5.2f}")
    return ecarts


def main(chemin: str) -> None:
    m1 = resample(load_mt5_csv(chemin), "1min")
    print(f"{len(m1):,} bougies M1 | seuil de force 0, sans prise partielle")
    print("Rappel : la detection s'arrete a la premiere confirmation trouvee,")
    print("dans l'ordre meche, englobante, CHoCH. Les groupes sont donc")
    print("construits par elimination et non independants.")

    tout = bloc(m1, "ECHANTILLON COMPLET", GRAINE)

    coupe = m1.index[len(m1) // 2]
    a = bloc(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100)
    b = bloc(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200)

    print("\n\nSTABILITE DANS LE TEMPS")
    print("  type            complet   1re moitie   2e moitie   stable ?")
    print("  " + "-" * 60)
    for nom in TYPES:
        if nom not in tout:
            continue
        va, vb = a.get(nom, np.nan), b.get(nom, np.nan)
        if np.isnan(va) or np.isnan(vb):
            verdict = "indeterminable"
        elif va > 0 and vb > 0:
            verdict = "positif des deux cotes"
        elif va * vb < 0:
            verdict = "S'INVERSE"
        else:
            verdict = "negatif des deux cotes"
        print(f"  {nom:<13} {tout[nom]:+.3f}     {va:+.3f}      {vb:+.3f}"
              f"     {verdict}")
    print("\n  Une confirmation n'est retenue que si son ecart reste du meme")
    print("  cote sur les DEUX moities. Un bon chiffre sur l'echantillon")
    print("  complet qui vient d'une seule periode ne se tradera pas.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
