"""Le critere de liquidite, isole.

POURQUOI LUI SEUL MERITE UNE MESURE DE PLUS
Des quatre criteres de qualification demandes, un seul produit un t
exploitable quand on compare les trades qui le respectent a ceux qui ne le
respectent pas : la presence d'un niveau evident juste au-dela de
l'extremite de la zone, la ou se trouve le stop. Les trades concernes
rapportent 0.087 R de moins sur l'echantillon complet (t = -1.57) et
0.121 R de moins sur la seconde moitie (t = -2.02). Le signe est celui
demande : mieux vaut que cette liquidite soit ABSENTE.

Les trois autres ne tiennent pas. L'imbalance donne un t de 0.68, 0.48 et
0.41 selon le bloc. L'impulsion propre donne 0.72, -0.21 et 1.36, et change
de signe. La position dans le mouvement donne un t de 4.85, mais avec le
signe INVERSE de celui demande, et les temoins separent autant que le reel :
c'est de la geometrie, pas de l'information.

CE QUI RESTE A VERIFIER
Le t de -2.02 vient d'une comparaison INTERNE a l'echantillon, qui ne
controle rien de la geometrie du systeme. Il faut donc le passer au meme
examen que tout le reste : applique en filtre a la detection, confronte a
des temoins, et mesure separement sur les deux moities. Et dans les deux
sens, puisque l'enseignement SMC affirme souvent le contraire.

Le troisieme essai mesure le biais de l'ordre limite : exiger que le prix
penetre la zone d'un spread avant de compter l'ordre rempli.

    python test_liq_seule.py chemin/vers/XAUUSD_M1.csv
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
BASE = dict(min_trend_force=0.0, partial_r=1e9, max_trades_per_day=50,
            entry_mode="contact", tp_usd=10.0)

CONFIGS = [
    ("sans aucun critere",          dict()),
    ("liquidite ABSENTE exigee",    dict(liq_rule="absente")),
    ("liquidite PRESENTE exigee",   dict(liq_rule="presente")),
    ("absente, tampon 0.25 ATR",    dict(liq_rule="absente", liq_buffer_atr=0.25)),
    ("absente, tampon 1.00 ATR",    dict(liq_rule="absente", liq_buffer_atr=1.0)),
    ("absente + marge de remplissage",
                                    dict(liq_rule="absente", fill_margin_usd=0.15)),
]


def mesure(m5: pd.DataFrame, m1: pd.DataFrame, opts: dict) -> dict:
    tr, st = backtest_mtf(m5, m1, MTFConfig(**BASE, **opts))
    if tr.empty:
        return {}
    return {"n": len(tr), "jour": st["trades_par_jour"],
            "wr": st["winrate"] * 100, "R": tr.R.mean(),
            "sd": tr.R.std(ddof=1), "stop": tr.stop_dist.median()}


def bloc(m1: pd.DataFrame, titre: str, graine: int) -> dict:
    m5 = resample(m1, "5min")
    jours = m5.index.normalize().nunique()
    print(f"\n{titre}  ({jours} jours)", flush=True)
    print("  configuration                     n   freq    WR   stop"
          "   R reel   temoin   ecart     t")
    print("  " + "-" * 84, flush=True)
    tem1 = [shuffle_bars(m1, seed=graine + k) for k in range(N_TEMOINS)]
    tem5 = [resample(t, "5min") for t in tem1]
    out = {}
    for nom, opts in CONFIGS:
        r = mesure(m5, m1, opts)
        if not r or r["n"] < 30:
            print(f"  {nom:<30} {r.get('n', 0):>5}  (trop peu de trades)",
                  flush=True)
            continue
        rs = [x for x in (mesure(t5, t1, opts)
                          for t5, t1 in zip(tem5, tem1)) if x]
        rt = float(np.mean([x["R"] for x in rs])) if rs else np.nan
        ec = r["R"] - rt
        t = ec / (r["sd"] / np.sqrt(r["n"])) if r["sd"] > 0 else np.nan
        out[nom] = (ec, r["jour"])
        print(f"  {nom:<30} {r['n']:>5} {r['jour']:>5.2f} {r['wr']:5.1f}% "
              f"{r['stop']:5.2f}  {r['R']:+.3f}  {rt:+.3f}  {ec:+.3f}  "
              f"{t:+5.2f}", flush=True)
    return out


def main(chemin: str) -> None:
    t0 = time.time()
    m1 = resample(load_mt5_csv(chemin), "1min")
    print(f"{len(m1):,} bougies M1 | entree au contact, cible 100 pips")
    print("Le critere de liquidite est le seul des quatre a produire un t")
    print("exploitable en comparaison interne. Il est ici applique en filtre,")
    print("seul, confronte aux temoins, et dans les deux sens.")

    tout = bloc(m1, "ECHANTILLON COMPLET", GRAINE)
    coupe = m1.index[len(m1) // 2]
    a = bloc(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100)
    b = bloc(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200)

    print("\n\nSTABILITE")
    print("  configuration                  complet    1re      2e     freq"
          "    verdict")
    print("  " + "-" * 80)
    ref = tout.get("sans aucun critere", (np.nan,))[0]
    for nom, _ in CONFIGS:
        if nom not in tout:
            continue
        nt, fr = tout[nom]
        va = a.get(nom, (np.nan,))[0]
        vb = b.get(nom, (np.nan,))[0]
        if np.isnan(va) or np.isnan(vb):
            v = "indeterminable"
        elif va > 0 and vb > 0:
            v = "POSITIF DES DEUX COTES"
        elif va * vb < 0:
            v = "s'inverse"
        else:
            v = "negatif des deux cotes"
        print(f"  {nom:<30} {nt:+.3f}   {va:+.3f}  {vb:+.3f}  {fr:5.2f}/j  {v}")

    print(f"\n  Reference sans aucun critere : ecart {ref:+.3f} R. Un filtre")
    print("  n'apporte quelque chose que s'il fait MIEUX que cette reference,")
    print("  garde le meme sens sur les deux moities, et conserve assez de")
    print("  trades pour etre tradable. Les trois conditions, pas une.")
    print(f"\n  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
