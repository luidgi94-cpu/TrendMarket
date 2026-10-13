"""L'avantage de H1 vient-il de l'Order Block, ou d'autre chose ?

CE QUI EST ETABLI
La detection en H1, sans aucun filtre de qualification, est la seule
configuration du projet a tenir le decoupage temporel avec les deux temoins
a la fois :

                    reel    temoin bars   ecart    temoin blocs   ecart
  echantillon      +0.440      +0.277     +0.163      +0.317     +0.123
  premiere moitie  +0.565      +0.398     +0.167      +0.334     +0.231
  seconde moitie   +0.324      +0.223     +0.101      +0.193     +0.131

Positif six fois sur six, avec des t de 2.3 a 3.4. Rien d'autre n'avait
survecu a cela.

LE PROBLEME QUE CE SCRIPT REGLE
Un temoin par blocs detruit l'ORDRE des bougies H1. Il detruit donc deux
choses en meme temps : la structure que pretend lire un Order Block, et la
persistance de la tendance. Un ecart positif dit seulement que l'ordre des
bougies H1 compte. Il ne dit pas lequel des deux ingredients compte.

QUATRE VARIANTES, TOUT LE RESTE IDENTIQUE
  vrai OB            la detection telle qu'elle est
  placebo            AUCUN Order Block : la zone est posee a 1.80 ATR en
                     arriere de la fin de l'impulsion, epaisseur 0.90 ATR,
                     deux valeurs reglees sur les medianes mesurees des
                     vraies zones. Meme tendance, meme impulsion, meme
                     entree au contact, meme stop, meme cible.
  tendance au sort   le vrai Order Block, mais la direction est tiree au
                     sort par tranche de quarante-huit bougies
  ni l'un ni l'autre le plancher de pure geometrie

Si le placebo fait aussi bien que le vrai Order Block, alors l'Order Block
n'apporte rien et il faut cesser de le chercher. Si la tendance au sort fait
aussi bien, c'est le filtre de tendance qui ne sert a rien. Si le plancher
fait aussi bien que tout le reste, l'avantage n'est ni dans l'un ni dans
l'autre et il faudra chercher dans la mecanique d'execution.

    python test_h1_decompose.py chemin/vers/XAUUSD_M1.csv
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
GRAINES_SORT = (7, 23, 91)        # plusieurs tirages, pour ne pas dependre d'un seul

VARIANTES = [
    ("vrai OB",            dict()),
    ("placebo, aucun OB",  dict(ob_mode="placebo")),
    ("tendance au sort",   dict(trend_mode="aleatoire")),
    ("ni l'un ni l'autre", dict(ob_mode="placebo", trend_mode="aleatoire")),
]


def config(opts: dict, seed: int = 0) -> MTFConfig:
    return MTFConfig(min_trend_force=0.0, partial_r=1e9, max_trades_per_day=50,
                     entry_mode="contact", tp_usd=10.0,
                     tf_minutes=60, atr_bars=24, max_hold_min=48 * 60,
                     trend_seed=seed, **opts)


def mesure(m1: pd.DataFrame, opts: dict) -> dict:
    """Moyenne sur plusieurs tirages si la direction est tiree au sort."""
    graines = GRAINES_SORT if opts.get("trend_mode") == "aleatoire" else (0,)
    res = []
    det = resample(m1, "1h")
    for s in graines:
        tr, st = backtest_mtf(det, m1, config(opts, s))
        if not tr.empty:
            res.append({"n": len(tr), "wr": st["winrate"] * 100,
                        "R": tr.R.mean(), "sd": tr.R.std(ddof=1),
                        "stop": tr.stop_dist.median()})
    if not res:
        return {}
    return {k: float(np.mean([r[k] for r in res])) for k in res[0]}


def bloc(m1: pd.DataFrame, titre: str, graine: int) -> dict:
    jours = m1.index.normalize().nunique()
    print(f"\n{titre}  ({jours} jours)", flush=True)
    print("  variante                 n    WR    stop    R reel   temoin"
          "   ecart     t")
    print("  " + "-" * 74, flush=True)
    tem = [shuffle_blocks(m1, 60, seed=graine + k) for k in range(N_TEMOINS)]
    out = {}
    for nom, opts in VARIANTES:
        r = mesure(m1, opts)
        if not r or r["n"] < 30:
            print(f"  {nom:<22} {r.get('n', 0):>5}  (trop peu de trades)",
                  flush=True)
            continue
        rs = [x for x in (mesure(t, opts) for t in tem) if x]
        rt = float(np.mean([x["R"] for x in rs])) if rs else np.nan
        ec = r["R"] - rt
        t = ec / (r["sd"] / np.sqrt(r["n"])) if r["sd"] > 0 else np.nan
        out[nom] = ec
        print(f"  {nom:<22} {r['n']:>5.0f} {r['wr']:5.1f}% {r['stop']:6.2f} "
              f"{r['R']:+.3f}  {rt:+.3f}  {ec:+.3f}  {t:+5.2f}", flush=True)
    return out


def main(chemin: str) -> None:
    t0 = time.time()
    m1 = resample(load_mt5_csv(chemin), "1min")
    print(f"{len(m1):,} bougies M1 | detection H1, gestion M1, "
          f"entree au contact, cible 100 pips")
    print("Temoin par blocs de soixante minutes : les bougies H1 du temoin")
    print("sont de vraies bougies H1, seul leur ordre est detruit.")

    tout = bloc(m1, "ECHANTILLON COMPLET", GRAINE)
    coupe = m1.index[len(m1) // 2]
    a = bloc(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100)
    b = bloc(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200)

    print("\n\nCE QUE CHAQUE INGREDIENT APPORTE")
    print("  variante               ecart complet    1re      2e     verdict")
    print("  " + "-" * 72)
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
        print(f"  {nom:<22} {tout[nom]:+.3f}        {va:+.3f}  {vb:+.3f}   {v}")

    base = tout.get("vrai OB", np.nan)
    pla = tout.get("placebo, aucun OB", np.nan)
    sort = tout.get("tendance au sort", np.nan)
    rien = tout.get("ni l'un ni l'autre", np.nan)
    print("\nLECTURE")
    print(f"  apport propre de l'Order Block  {base - pla:+.3f} R"
          f"   (vrai OB moins placebo)")
    print(f"  apport propre de la tendance    {base - sort:+.3f} R"
          f"   (vrai OB moins tirage au sort)")
    print(f"  plancher de pure geometrie      {rien:+.3f} R")
    print("  Si le plancher est deja positif et proche du vrai OB, l'avantage")
    print("  n'est ni dans la zone ni dans la tendance : il est dans la")
    print("  mecanique, et une mecanique ne se trade pas, elle se corrige.")
    print(f"\n  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
