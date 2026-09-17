"""Corps seul et deduplication changent-ils quelque chose ?

Deux modifications viennent d'etre apportees a l'indicateur pour que ses
zones cessent de ressembler a un bloc de bougies :

  CORPS SEUL     la zone va du haut au bas du CORPS de la bougie inverse,
                 au lieu d'aller de sa meche haute a sa meche basse
  DEDUPLICATION  une nouvelle zone recouvrant une zone existante de meme
                 sens au-dela d'un seuil n'est pas creee

La premiere etait presentee comme un choix d'affichage, mais elle ne l'est
pas : une zone plus fine place le stop plus pres, donc change le R de
chaque trade. La seconde agit franchement sur la detection et retire des
signaux. Les deux doivent donc etre mesurees et non supposees neutres.

CE QUE L'ON ATTEND, ET POURQUOI IL FAUT S'EN MEFIER
Une zone plus fine produit mecaniquement un R plus eleve : a mouvement
egal, un stop deux fois plus court double le R. Cela ne signale aucun gain
d'information, et le temoin en profite exactement autant. Seul l'ECART au
temoin est lu. La deduplication, elle, doit etre a peu pres neutre si les
zones ecartees sont bien des quasi-doublons ; un ecart qui bougerait
beaucoup signalerait qu'elle retire autre chose que des doublons.

    python test_zone_fine.py chemin/vers/XAUUSD_M1.csv
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

VARIANTES = [
    ("meches + doublons", False, 1.0),   # comportement d'origine
    ("meches + dedup",    False, 0.5),
    ("CORPS + doublons",  True,  1.0),
    ("CORPS + dedup",     True,  0.5),   # nouveau defaut de l'indicateur
]


def mesure(m1: pd.DataFrame, use_body: bool, dedup: float) -> dict:
    m5 = resample(m1, "5min")
    cfg = MTFConfig(min_trend_force=0.0, partial_r=1e9,
                    use_body=use_body, dedup_overlap=dedup)
    tr, st = backtest_mtf(m5, m1, cfg)
    if tr.empty:
        return {}
    return {"n": len(tr), "jour": st["trades_par_jour"],
            "wr": (tr.pnl > 0).mean() * 100, "R": tr.R.mean(),
            "sd": tr.R.std(ddof=1), "stop": tr.stop_dist.median()}


def ligne(m1, temoins, nom, body, dedup) -> tuple:
    r = mesure(m1, body, dedup)
    if not r:
        print(f"  {nom:<20} aucun trade")
        return nom, np.nan
    rs = [mesure(t1, body, dedup) for t1 in temoins]
    rs = [x for x in rs if x]
    rt = float(np.mean([x["R"] for x in rs])) if rs else np.nan
    ec = r["R"] - rt
    t = ec / (r["sd"] / np.sqrt(r["n"])) if r["sd"] > 0 else np.nan
    print(f"  {nom:<20} {r['n']:>5} {r['jour']:>5.2f}/j  WR {r['wr']:4.1f}%  "
          f"stop {r['stop']:4.2f}  R {r['R']:+.3f}  temoin {rt:+.3f}  "
          f"ecart {ec:+.3f}  t {t:+5.2f}")
    return nom, ec


def main(chemin: str) -> None:
    m1 = resample(load_mt5_csv(chemin), "1min")
    print(f"{len(m1):,} bougies M1 | "
          f"{resample(m1,'5min').index.normalize().nunique()} jours")
    print("seuil de force 0, sans prise partielle\n")

    print(f"Construction de {N_TEMOINS} temoins...")
    temoins = [shuffle_bars(m1, seed=GRAINE + k) for k in range(N_TEMOINS)]
    print("fait.\n")

    print("  variante                 n  freq    winrate  stop   R reel"
          "   temoin   ecart     t")
    print("  " + "-" * 92)
    res = {}
    for nom, body, dedup in VARIANTES:
        n, ec = ligne(m1, temoins, nom, body, dedup)
        res[n] = ec

    print("\nLECTURE")
    base = res.get("meches + doublons", np.nan)
    neuf = res.get("CORPS + dedup", np.nan)
    print(f"  Ancien defaut  (meches, doublons) : ecart {base:+.3f} R")
    print(f"  Nouveau defaut (corps,  dedup)    : ecart {neuf:+.3f} R")
    if not np.isnan(base) and not np.isnan(neuf):
        print(f"  Difference : {neuf - base:+.3f} R")
        print("\n  Le R brut monte forcement avec le corps seul, puisque le")
        print("  stop raccourcit. Cela ne vaut pas gain d'information : le")
        print("  temoin monte pour la meme raison. Seule la colonne 'ecart'")
        print("  repond, et aucune valeur ne devient exploitable si elle")
        print("  n'atteint pas t = 2 ET ne survit pas hors echantillon.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
