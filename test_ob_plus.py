"""OB+ vaut-il mieux qu'un OB simple ?

L'indicateur distingue depuis le debut deux categories de zones :

    OB+  l'Order Block est suivi d'une imbalance : la troisieme bougie du
         mouvement ne revient pas combler son extremite
    OB   pas d'imbalance

Cette distinction structure l'affichage et le tableau depuis le premier
jour, mais elle n'a jamais ete mesuree. Elle est tenue pour acquise parce
qu'elle fait partie du corpus, ce qui n'est pas une raison.

LE CRITERE
Comme pour la confluence, il ne suffit pas que les OB+ affichent un R plus
eleve. Une zone suivie d'une imbalance est une zone situee dans un
mouvement violent : les stops y sont plus larges et les objectifs plus
lointains, donc tout rapporte davantage, le hasard compris. On compare
donc l'ECART AU TEMOIN des deux groupes, chaque groupe face a son propre
temoin classe de la meme facon.

    python test_ob_plus.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quant.mt5_loader import load_mt5_csv, resample
from quant.scalp_mtf import MTFConfig, backtest_mtf
from quant.surrogate import shuffle_bars

N_TEMOINS = 4
GRAINE = 20260916


def groupes(m1: pd.DataFrame, cfg: MTFConfig) -> dict:
    m5 = resample(m1, "5min")
    tr, _ = backtest_mtf(m5, m1, cfg)
    if tr.empty:
        return {}
    return {"OB+": tr.R[tr.ob_plus], "OB": tr.R[~tr.ob_plus], "tout": tr.R}


def main(chemin: str) -> None:
    m1 = resample(load_mt5_csv(chemin), "1min")
    cfg = MTFConfig(min_trend_force=0.8, partial_r=1e9)
    print(f"{len(m1):,} bougies M1 | seuil 0.8, sans prise partielle\n")

    print("Donnees reelles...")
    reel = groupes(m1, cfg)
    print(f"Temoins ({N_TEMOINS}, classement OB+/OB recalcule a l'identique)...")
    tem = {k: [] for k in reel}
    for k in range(N_TEMOINS):
        g = groupes(shuffle_bars(m1, seed=GRAINE + k), cfg)
        for nom, v in g.items():
            if len(v):
                tem[nom].append(v.mean())
    print("fait.\n")

    print("  groupe    n      WR      R reel   temoin    ecart      t")
    print("  " + "-" * 56)
    res = {}
    for nom in ("OB+", "OB", "tout"):
        v = reel.get(nom, pd.Series(dtype=float))
        if len(v) < 30:
            print(f"  {nom:<8} {len(v):>4}  (trop peu de trades)")
            continue
        rt = float(np.mean(tem[nom])) if tem[nom] else np.nan
        ec = v.mean() - rt
        t = ec / (v.std(ddof=1) / np.sqrt(len(v))) if v.std(ddof=1) > 0 else np.nan
        res[nom] = ec
        print(f"  {nom:<8} {len(v):>4}  {(v > 0).mean() * 100:5.1f}%  "
              f"{v.mean():+.3f}  {rt:+.3f}  {ec:+.3f}  {t:+6.2f}")

    print("\nVERDICT")
    if "OB+" in res and "OB" in res:
        d = res["OB+"] - res["OB"]
        print(f"  Ecart OB+ moins ecart OB : {d:+.3f} R")
        if d > 0.05:
            print("  L'imbalance semble apporter quelque chose SUR CET")
            print("  ECHANTILLON. A confirmer hors echantillon avant d'en")
            print("  faire un critere de selection.")
        elif d < -0.05:
            print("  Les OB+ font MOINS BIEN que les OB simples. La")
            print("  distinction est contre-productive.")
        else:
            print("  Aucune difference exploitable. L'imbalance est")
            print("  descriptive, pas predictive : elle decrit la violence du")
            print("  mouvement passe, elle ne dit rien du mouvement a venir.")
    print("\n  Rappel : le R brut des OB+ sera probablement plus eleve. Cela")
    print("  ne prouve rien tant que le temoin monte autant, ce qui signale")
    print("  une selection de volatilite et non d'information.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
