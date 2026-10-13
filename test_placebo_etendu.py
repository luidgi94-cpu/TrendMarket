"""Deux controles qui decident du sort du resultat H1.

OU EN EST-ON
La zone posee par geometrie, sans aucun Order Block, donne un ecart au
temoin positif sur les quinze reglages essayes, et sur les DEUX moities a
chaque fois. La surface est lisse et orientee : l'ecart croit quand le recul
diminue et quand la zone s'amincit, dans le meme sens sur les deux moities.
Rien d'autre dans ce projet n'avait tenu cela.

                 0.60     0.90     1.20      <- epaisseur, en ATR
       1.20    +0.250   +0.221   +0.189
       1.50    +0.248   +0.228   +0.141
       1.80    +0.248   +0.160   +0.126
       2.10    +0.150   +0.131   +0.123
       2.50    +0.045   +0.072   +0.066

PREMIER CONTROLE : L'OPTIMUM EST-IL DEHORS ?
Le meilleur couple se trouve dans le COIN de la grille, recul 1.20 et
epaisseur 0.60, tous deux au bord. Une surface qui monte encore en sortant
du cadre ne dit pas ou est son sommet. La grille est donc prolongee vers les
reculs plus courts et les zones plus minces. Si le sommet apparait a
l'interieur du nouveau cadre, le reglage est un vrai optimum. S'il fuit
encore vers le bord, l'effet est mecanique : amincir la zone raccourcit le
stop, et une cible fixe en dollars vaut alors davantage de R, ce qui monte
le R sans rien apporter.

SECOND CONTROLE, LE PLUS IMPORTANT : LA CIBLE
Toute cette serie utilise une cible FIXE de 100 pips. Si l'avantage vient
vraiment d'un suivi de tendance avec entree sur repli, il doit survivre a
une cible proportionnelle au risque, qui ne depend plus du tout de
l'epaisseur de la zone. S'il disparait, alors l'avantage n'etait pas dans la
lecture du marche mais dans le rapport entre une cible fixe et un stop
variable, c'est-a-dire dans la forme du pari.

    python test_placebo_etendu.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from quant.mt5_loader import load_mt5_csv, resample
from quant.scalp_mtf import MTFConfig, backtest_mtf
from quant.surrogate import shuffle_blocks

N_TEMOINS = 4
GRAINE = 20261013
JAMBES = [0.6, 0.8, 1.0, 1.2]
ZONES = [0.20, 0.30, 0.45, 0.60]
# Reglage de reference pour l'essai sur la cible : le meilleur de la grille
# precedente, qui reste le mieux documente.
REF = dict(placebo_jambe_atr=1.2, placebo_zone_atr=0.6)
CIBLES = [("fixe 100 pips", dict(tp_usd=10.0)),
          ("fixe  50 pips", dict(tp_usd=5.0)),
          ("1.5 R", dict(tp_usd=0.0, tp_r=1.5)),
          ("2.0 R", dict(tp_usd=0.0, tp_r=2.0)),
          ("3.0 R", dict(tp_usd=0.0, tp_r=3.0))]


def cfg(opts: dict) -> MTFConfig:
    base = dict(min_trend_force=0.0, partial_r=1e9, max_trades_per_day=50,
                entry_mode="contact", tp_usd=10.0,
                tf_minutes=60, atr_bars=24, max_hold_min=48 * 60,
                ob_mode="placebo")
    base.update(opts)
    return MTFConfig(**base)


def mesure(m1: pd.DataFrame, det: pd.DataFrame, opts: dict) -> dict:
    tr, st = backtest_mtf(det, m1, cfg(opts))
    if tr.empty:
        return {}
    return {"n": len(tr), "jour": st["trades_par_jour"],
            "wr": st["winrate"] * 100, "R": tr.R.mean(),
            "sd": tr.R.std(ddof=1), "stop": tr.stop_dist.median(),
            "tp": tr.tp_en_r.median()}


def serie(m1: pd.DataFrame, titre: str, graine: int,
          essais: list) -> dict:
    print(f"\n{titre}", flush=True)
    print("  essai                      n   freq    WR   stop    TP"
          "    R reel   temoin   ecart     t")
    print("  " + "-" * 88, flush=True)
    det = resample(m1, "1h")
    tem = [shuffle_blocks(m1, 60, seed=graine + k) for k in range(N_TEMOINS)]
    tem_det = [resample(t, "1h") for t in tem]
    out = {}
    for nom, opts in essais:
        r = mesure(m1, det, opts)
        if not r or r["n"] < 30:
            print(f"  {nom:<24} {r.get('n', 0):>5}  (trop peu de trades)",
                  flush=True)
            continue
        rs = [x for x in (mesure(t, td, opts)
                          for t, td in zip(tem, tem_det)) if x]
        rt = float(np.mean([x["R"] for x in rs])) if rs else np.nan
        ec = r["R"] - rt
        t = ec / (r["sd"] / np.sqrt(r["n"])) if r["sd"] > 0 else np.nan
        out[nom] = ec
        print(f"  {nom:<24} {r['n']:>5} {r['jour']:>5.2f} {r['wr']:5.1f}% "
              f"{r['stop']:5.2f} {r['tp']:5.2f}R {r['R']:+.3f}  {rt:+.3f}  "
              f"{ec:+.3f}  {t:+5.2f}", flush=True)
    return out


def verdict(titre: str, tout: dict, a: dict, b: dict) -> None:
    print(f"\n{titre}")
    print("  essai                    complet    1re      2e     verdict")
    print("  " + "-" * 70)
    for nom in tout:
        va, vb = a.get(nom, np.nan), b.get(nom, np.nan)
        if np.isnan(va) or np.isnan(vb):
            v = "indeterminable"
        elif va > 0 and vb > 0:
            v = "POSITIF DES DEUX COTES"
        elif va * vb < 0:
            v = "s'inverse"
        else:
            v = "negatif des deux cotes"
        print(f"  {nom:<24} {tout[nom]:+.3f}   {va:+.3f}  {vb:+.3f}   {v}")


def main(chemin: str) -> None:
    t0 = time.time()
    m1 = resample(load_mt5_csv(chemin), "1min")
    coupe = m1.index[len(m1) // 2]
    m1a, m1b = m1[m1.index < coupe], m1[m1.index >= coupe]
    print(f"{len(m1):,} bougies M1 | detection H1, zone posee par geometrie")

    print("\n\n=========  PROLONGEMENT DE LA GRILLE  =========")
    grille = [(f"recul {j:.1f}  ep. {z:.2f}",
               dict(placebo_jambe_atr=j, placebo_zone_atr=z))
              for j in JAMBES for z in ZONES]
    g_tout = serie(m1, "ECHANTILLON COMPLET", GRAINE, grille)
    g_a = serie(m1a, "PREMIERE MOITIE", GRAINE + 100, grille)
    g_b = serie(m1b, "SECONDE MOITIE", GRAINE + 200, grille)

    print("\n  CARTE DE L'ECART, echantillon complet")
    print("  recul \\ epaiss." + "".join(f"{z:>9.2f}" for z in ZONES))
    for j in JAMBES:
        ligne = f"  {j:>14.1f}"
        for z in ZONES:
            v = g_tout.get(f"recul {j:.1f}  ep. {z:.2f}", np.nan)
            ligne += "      --- " if np.isnan(v) else f"{v:>+9.3f}"
        print(ligne)
    verdict("STABILITE DU PROLONGEMENT", g_tout, g_a, g_b)

    print("\n\n=========  LA CIBLE  =========")
    print("Meme zone (recul 1.2 ATR, epaisseur 0.6 ATR), seule la cible change.")
    essais = [(nom, dict(REF, **o)) for nom, o in CIBLES]
    c_tout = serie(m1, "ECHANTILLON COMPLET", GRAINE, essais)
    c_a = serie(m1a, "PREMIERE MOITIE", GRAINE + 100, essais)
    c_b = serie(m1b, "SECONDE MOITIE", GRAINE + 200, essais)
    verdict("STABILITE SELON LA CIBLE", c_tout, c_a, c_b)

    print("\nLECTURE")
    print("  Si l'avantage ne tient qu'avec une cible fixe en dollars, il")
    print("  n'est pas une lecture du marche mais le produit d'un rapport")
    print("  entre une cible fixe et un stop variable. S'il tient avec une")
    print("  cible proportionnelle au risque, il est independant de la")
    print("  geometrie et c'est alors un vrai suivi de tendance.")
    print(f"\n  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
