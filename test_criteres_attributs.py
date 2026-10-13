"""Chaque critere separe-t-il les trades, independamment du filtre ?

POURQUOI CE SECOND TEST
Appliquer un critere a la detection fait deux choses a la fois : il ecarte
des zones, et il change l'enchainement, puisque refuser une zone libere la
place pour la suivante. Quand le resultat se degrade, on ne sait pas si
c'est parce que le critere designe de mauvaises zones ou parce qu'il casse
la sequence.

Ici, aucun critere ne filtre. La population complete est tradee, chaque
critere est ENREGISTRE pour chaque trade, et l'on compare le R des trades
qui le respectent a celui des trades qui ne le respectent pas. Le filtre et
l'attribut repondent a deux questions differentes, et il faut les deux :

  le FILTRE     dit si le systeme devient meilleur quand on l'applique
  l'ATTRIBUT    dit si le critere designe effectivement de meilleures zones

Un critere peut etre vrai comme attribut et inutilisable comme filtre, et
c'est exactement ce qui est arrive a la forme de la bougie d'Order Block.

CE QUI EST COMPARE
  ob_plus    l'Order Block genere une imbalance            0 / 1
  a_propre   aucune meche ne revient dans la zone          0 / 1
  a_liq      un niveau evident se trouve juste au-dela     0 / 1
  a_fib      position de la zone dans le mouvement, de 0
             (a l'origine du mouvement, donc en discount
             pour un achat) a 1 (a son extremite)       quintiles

Chaque ecart est confronte a celui des temoins : un critere qui separe des
bougies melangees separe par geometrie et non par information.

    python test_criteres_attributs.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from quant.mt5_loader import load_mt5_csv, resample
from quant.scalp_mtf import MTFConfig, backtest_mtf
from quant.surrogate import shuffle_bars

N_TEMOINS = 4
GRAINE = 20261013
NQ = 5

BINAIRES = {
    "ob_plus":  "imbalance generee",
    "a_propre": "impulsion propre",
    "a_liq":    "liquidite juste au-dela",
}


def trades(m1: pd.DataFrame) -> pd.DataFrame:
    cfg = MTFConfig(min_trend_force=0.0, partial_r=1e9, max_trades_per_day=50,
                    entry_mode="contact", tp_usd=10.0)
    tr, _ = backtest_mtf(resample(m1, "5min"), m1, cfg)
    return tr


def ecart_binaire(tr: pd.DataFrame, col: str) -> tuple:
    v = tr[[col, "R"]].dropna()
    g0 = v.loc[v[col] == 0, "R"]
    g1 = v.loc[v[col] == 1, "R"]
    if len(g0) < 30 or len(g1) < 30:
        return np.nan, np.nan, len(g0), len(g1)
    return g1.mean() - g0.mean(), g0.mean(), len(g0), len(g1)


def ecart_quintiles(tr: pd.DataFrame, col: str) -> tuple:
    v = tr[[col, "R"]].dropna()
    if len(v) < 100 or v[col].nunique() < NQ:
        return np.nan, np.nan, None
    try:
        q = pd.qcut(v[col], NQ, labels=False, duplicates="drop")
    except ValueError:
        return np.nan, np.nan, None
    moy = v.groupby(q).R.mean()
    if len(moy) < 3:
        return np.nan, np.nan, None
    rho = np.corrcoef(np.arange(len(moy)), moy.to_numpy())[0, 1]
    return moy.iloc[-1] - moy.iloc[0], rho, moy


def bloc(m1: pd.DataFrame, titre: str, graine: int) -> dict:
    print(f"\n{titre}", flush=True)
    reel = trades(m1)
    if reel.empty:
        print("  aucun trade")
        return {}
    tems = [t for t in (trades(shuffle_bars(m1, seed=graine + k))
                        for k in range(N_TEMOINS)) if not t.empty]
    print(f"  {len(reel)} trades reels, {len(tems)} temoins", flush=True)

    out = {}
    print("  critere                         n=0    n=1    R(0)    R(1)"
          "   ecart  temoin    net")
    print("  " + "-" * 82)
    for col, nom in BINAIRES.items():
        ec, r0, n0, n1 = ecart_binaire(reel, col)
        if np.isnan(ec):
            print(f"  {nom:<28}  (groupes trop deseequilibres : "
                  f"{n0} / {n1})")
            continue
        ecs = [ecart_binaire(t, col)[0] for t in tems]
        ecs = [x for x in ecs if not np.isnan(x)]
        et = float(np.mean(ecs)) if ecs else np.nan
        out[nom] = ec - et
        print(f"  {nom:<28} {n0:>6} {n1:>6} {r0:+.3f} {r0 + ec:+.3f} "
              f"{ec:+.3f}  {et:+.3f}  {ec - et:+.3f}", flush=True)

    ec, rho, moy = ecart_quintiles(reel, "a_fib")
    if not np.isnan(ec):
        ecs = [ecart_quintiles(t, "a_fib")[0] for t in tems]
        ecs = [x for x in ecs if not np.isnan(x)]
        et = float(np.mean(ecs)) if ecs else np.nan
        out["position dans le mouvement"] = ec - et
        print(f"\n  position dans le mouvement, par quintile "
              f"(Q1 = le plus en discount)")
        print("    " + "  ".join(f"Q{k + 1} {v:+.3f}"
                                 for k, v in enumerate(moy.to_numpy())))
        print(f"    ecart Q5-Q1 {ec:+.3f}   temoin {et:+.3f}   "
              f"net {ec - et:+.3f}   rho {rho:+.2f}", flush=True)
    return out


def main(chemin: str) -> None:
    t0 = time.time()
    m1 = resample(load_mt5_csv(chemin), "1min")
    print(f"{len(m1):,} bougies M1 | entree au contact, cible 100 pips")
    print("AUCUN critere ne filtre : la population complete est tradee et")
    print("chaque critere est seulement enregistre. On mesure donc si le")
    print("critere designe de meilleures zones, pas s'il ameliore le systeme.")

    tout = bloc(m1, "ECHANTILLON COMPLET (436 jours)", GRAINE)
    coupe = m1.index[len(m1) // 2]
    a = bloc(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100)
    b = bloc(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200)

    print("\n\nCE QUI TIENT SUR LES DEUX MOITIES")
    print("  critere                       net complet    1re      2e"
          "     verdict")
    print("  " + "-" * 78)
    for nom in list(BINAIRES.values()) + ["position dans le mouvement"]:
        if nom not in tout:
            continue
        nt = tout[nom]
        va, vb = a.get(nom, np.nan), b.get(nom, np.nan)
        if np.isnan(va) or np.isnan(vb):
            v = "indeterminable"
        elif abs(va) < 0.05 and abs(vb) < 0.05:
            v = "plat des deux cotes"
        elif va * vb > 0:
            v = "MEME SENS"
        else:
            v = "s'inverse"
        print(f"  {nom:<28} {nt:+.3f}    {va:+.3f}  {vb:+.3f}   {v}")

    print("\n  Le signe compte autant que la taille. Pour l'imbalance et")
    print("  l'impulsion propre, un net POSITIF confirme le critere. Pour la")
    print("  liquidite juste au-dela, un net NEGATIF le confirme : c'est bien")
    print("  son absence qu'il faudrait exiger. Pour la position, un net")
    print("  NEGATIF confirme le discount, puisque Q1 est le plus profond.")
    print(f"\n  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
