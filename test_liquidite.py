"""Le balayage de liquidite porte-t-il de l'information ?

L'HYPOTHESE, TELLE QU'ELLE EST ENSEIGNEE
Des ordres stop s'accumulent au-dela des extremes evidents : plus haut et
plus bas de la veille, sommets et creux recents. Celui qui doit executer
une taille importante a interet a les declencher, parce qu'ils lui
fournissent la contrepartie. Le prix depasse donc brievement le niveau,
absorbe les ordres, puis repart dans l'autre sens.

CE QUI EST TESTABLE ET CE QUI NE L'EST PAS
Le balayage lui-meme est falsifiable : un depassement suivi d'un retour
dans l'ancienne fourchette doit produire davantage qu'un temoin ou les
bougies ont ete melangees. C'est ce que mesure ce script.

La PROVENANCE de la liquidite, retail ou institutionnelle, ne l'est pas.
Des donnees OHLC ne contiennent aucune information sur l'identite de la
contrepartie : un carnet d'ordres ne se deduit pas de quatre prix par
bougie. Toute affirmation sur "qui" a pris la liquidite est donc
invérifiable avec ce que nous avons, et n'est pas testee ici.

PROTOCOLE
Chaque variante est confrontee a des temoins construits en melangeant les
bougies M1 a l'interieur de chaque heure, et le decoupage temporel est
applique d'emblee : l'echantillon complet ne sert qu'a situer, les deux
moities decident.

    python test_liquidite.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quant.mt5_loader import load_mt5_csv, resample
from quant.liquidite import LiqConfig, backtest_liquidite
from quant.surrogate import shuffle_bars

N_TEMOINS = 4
GRAINE = 20260924

# (etiquette, niveaux utilises, part minimale de meche, filtre de tendance)
VARIANTES = [
    ("veille seule",        dict(use_pdh_pdl=True,  use_pivots=False)),
    ("pivots seuls",        dict(use_pdh_pdl=False, use_pivots=True)),
    ("les deux",            dict(use_pdh_pdl=True,  use_pivots=True)),
    ("les deux, rejet net", dict(use_pdh_pdl=True,  use_pivots=True,
                                 min_wick_frac=0.6, max_close_back=0.7)),
    ("les deux, dans la tendance",
                            dict(use_pdh_pdl=True,  use_pivots=True, use_trend=True)),
]


def mesure(m1: pd.DataFrame, opts: dict) -> dict:
    cfg = LiqConfig(**opts)
    tr, st = backtest_liquidite(resample(m1, "5min"), cfg)
    if tr.empty:
        return {}
    return {"n": len(tr), "jour": st["trades_par_jour"],
            "wr": st["winrate"] * 100, "R": tr.R.mean(),
            "sd": tr.R.std(ddof=1), "stop": tr.stop_dist.median()}


def bloc(m1: pd.DataFrame, titre: str, graine: int) -> dict:
    jours = resample(m1, "5min").index.normalize().nunique()
    print(f"\n{titre}  ({jours} jours)")
    print("  variante                       n  freq   WR     stop"
          "   R reel   temoin   ecart     t")
    print("  " + "-" * 82)
    temoins = [shuffle_bars(m1, seed=graine + k) for k in range(N_TEMOINS)]
    out = {}
    for nom, opts in VARIANTES:
        r = mesure(m1, opts)
        if not r or r["n"] < 30:
            print(f"  {nom:<28} {r.get('n', 0):>5}  (trop peu de trades)")
            continue
        rs = [mesure(t1, opts) for t1 in temoins]
        rs = [x for x in rs if x]
        rt = float(np.mean([x["R"] for x in rs])) if rs else np.nan
        ec = r["R"] - rt
        t = ec / (r["sd"] / np.sqrt(r["n"])) if r["sd"] > 0 else np.nan
        out[nom] = ec
        print(f"  {nom:<28} {r['n']:>5} {r['jour']:>5.2f} {r['wr']:4.1f}% "
              f"{r['stop']:5.2f}  {r['R']:+.3f}  {rt:+.3f}  {ec:+.3f}  {t:+5.2f}")
    return out


def main(chemin: str) -> None:
    m1 = resample(load_mt5_csv(chemin), "1min")
    print(f"{len(m1):,} bougies M1 | stop au-dela de l'extreme du balayage, cible 3 R")
    print("La provenance de la liquidite, retail ou institutionnelle, n'est pas")
    print("testable sur des donnees OHLC et n'est donc pas mesuree ici.")

    tout = bloc(m1, "ECHANTILLON COMPLET", GRAINE)
    coupe = m1.index[len(m1) // 2]
    a = bloc(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100)
    b = bloc(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200)

    print("\n\nSTABILITE")
    print("  variante                      complet   1re moitie   2e moitie   verdict")
    print("  " + "-" * 78)
    for nom, _ in VARIANTES:
        if nom not in tout:
            continue
        va, vb = a.get(nom, np.nan), b.get(nom, np.nan)
        if np.isnan(va) or np.isnan(vb):
            v = "indeterminable"
        elif va > 0 and vb > 0:
            v = "POSITIF DES DEUX COTES"
        elif va * vb < 0:
            v = "s'inverse"
        else:
            v = "negatif des deux cotes"
        print(f"  {nom:<28} {tout[nom]:+.3f}     {va:+.3f}      {vb:+.3f}     {v}")
    print("\n  Une variante n'est retenue que si son ecart reste du meme cote sur")
    print("  les DEUX moities. C'est le critere qui a elimine toutes les")
    print("  configurations d'Order Block testees jusqu'ici.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
