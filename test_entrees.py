"""Confirmer l'entree sert-il a quelque chose ?

L'utilisateur fait remarquer, a juste titre, que le corpus compte bien
plus de trois signaux d'entree. Avant d'en ajouter vingt, il faut poser
la question dans le bon ordre.

POURQUOI NE PAS SE CONTENTER D'EN TESTER DAVANTAGE
Chaque type teste est un tirage. A vingt tirages, la probabilite d'en
voir au moins un depasser t = 2 par pur hasard avoisine une chance sur
deux, meme si aucun ne vaut rien. Multiplier les candidats ne rapproche
pas de la verite, cela garantit un faux positif.

LA QUESTION PREALABLE
Le cas de base n'a jamais ete mesure : entrer des que le prix TOUCHE la
zone, sans rien attendre. Si l'entree au toucher fait aussi bien que les
entrees confirmees, alors tout l'appareillage de confirmations est
decoratif, et en tester vingt de plus est du temps perdu. Si elle fait
nettement moins bien, alors confirmer apporte quelque chose et il devient
legitime de chercher la meilleure facon de le faire.

CORRECTION D'UN DEFAUT DU TEST PRECEDENT
Les trois types y etaient mesures PAR ELIMINATION : la fonction
s'arretant a la premiere confirmation trouvee, un trade classe CHoCH
etait un trade ou ni meche ni englobante n'etaient presentes. On ne
comparait donc pas trois signaux mais trois restes. Chaque type est ici
active SEUL.

DEUX TYPES AJOUTES
    pinbar        variante stricte de la meche, corps dans le tiers oppose
    inside_break  sortie d'une bougie contenue dans la precedente

    python test_entrees.py chemin/vers/XAUUSD_M1.csv
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

# (etiquette, types actifs, exiger une confirmation)
MODES = [
    ("AUCUNE (au toucher)", (),                 False),
    ("meche_rejet seule",   ("meche_rejet",),   True),
    ("englobante seule",    ("englobante",),    True),
    ("choch seul",          ("choch",),         True),
    ("pinbar seul",         ("pinbar",),        True),
    ("inside_break seul",   ("inside_break",),  True),
    ("les trois d'origine", ("meche_rejet", "englobante", "choch"), True),
]


def mesure(m1: pd.DataFrame, types: tuple, exiger: bool) -> dict:
    cfg = MTFConfig(min_trend_force=0.0, partial_r=1e9,
                    confirm_types=types, require_confirm=exiger)
    tr, st = backtest_mtf(resample(m1, "5min"), m1, cfg)
    if tr.empty:
        return {}
    return {"n": len(tr), "jour": st["trades_par_jour"],
            "wr": (tr.pnl > 0).mean() * 100, "R": tr.R.mean(),
            "sd": tr.R.std(ddof=1)}


def bloc(m1: pd.DataFrame, titre: str, graine: int) -> dict:
    jours = resample(m1, "5min").index.normalize().nunique()
    print(f"\n{titre}  ({jours} jours)")
    print("  mode                     n   freq    WR      R reel"
          "   temoin   ecart     t")
    print("  " + "-" * 74)
    temoins = [shuffle_bars(m1, seed=graine + k) for k in range(N_TEMOINS)]
    out = {}
    for nom, types, exiger in MODES:
        r = mesure(m1, types, exiger)
        if not r or r["n"] < 30:
            print(f"  {nom:<22} {r.get('n', 0):>5}  (trop peu de trades)")
            continue
        rs = [mesure(t1, types, exiger) for t1 in temoins]
        rs = [x for x in rs if x]
        rt = float(np.mean([x["R"] for x in rs])) if rs else np.nan
        ec = r["R"] - rt
        t = ec / (r["sd"] / np.sqrt(r["n"])) if r["sd"] > 0 else np.nan
        out[nom] = ec
        print(f"  {nom:<22} {r['n']:>5} {r['jour']:>5.2f}/j {r['wr']:5.1f}%  "
              f"{r['R']:+.3f}  {rt:+.3f}  {ec:+.3f}  {t:+5.2f}")
    return out


def main(chemin: str) -> None:
    m1 = resample(load_mt5_csv(chemin), "1min")
    print(f"{len(m1):,} bougies M1 | seuil de force 0, sans prise partielle")
    print("Chaque type est active SEUL, et non par elimination.")

    tout = bloc(m1, "ECHANTILLON COMPLET", GRAINE)
    coupe = m1.index[len(m1) // 2]
    a = bloc(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100)
    b = bloc(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200)

    print("\n\nSTABILITE")
    print("  mode                    complet   1re moitie   2e moitie   verdict")
    print("  " + "-" * 72)
    for nom, _, _ in MODES:
        if nom not in tout:
            continue
        va, vb = a.get(nom, np.nan), b.get(nom, np.nan)
        if np.isnan(va) or np.isnan(vb):
            v = "indeterminable"
        elif va > 0 and vb > 0:
            v = "positif des deux cotes"
        elif va * vb < 0:
            v = "s'inverse"
        else:
            v = "negatif des deux cotes"
        print(f"  {nom:<22} {tout[nom]:+.3f}     {va:+.3f}      {vb:+.3f}"
              f"     {v}")

    base = tout.get("AUCUNE (au toucher)", np.nan)
    print(f"\n  Cas de base, entree au toucher : ecart {base:+.3f} R")
    if not np.isnan(base):
        mieux = [n for n, e in tout.items()
                 if n != "AUCUNE (au toucher)" and e > base]
        if not mieux:
            print("  AUCUNE confirmation ne fait mieux que d'entrer sans rien")
            print("  attendre. L'appareillage de confirmations ne sert alors")
            print("  a rien, et en tester d'autres ne servira pas davantage.")
        else:
            print(f"  Font mieux que le cas de base : {', '.join(mieux)}")
            print("  A ne retenir que si l'avantage tient sur LES DEUX moities.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
