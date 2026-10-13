"""La VWAP de seance et ses bandes portent-elles un avantage ?

LA STRATEGIE TESTEE
Celle de la capture : des zones publiees en haut et en bas de la fourchette
du jour, vendues en haut, achetees en bas, avec la moyenne comme cible. Mise
en regle : on vend la bande a plus k ecarts-types de la VWAP de seance, on
achete celle a moins k, et l'on vise la VWAP. C'est le pari du retour a la
moyenne, et c'est l'usage que les formateurs font de cet outil.

LE MOTEUR EST VALIDE AVANT D'ETRE CRU
Sur six marches aleatoires sans derive, ou l'esperance doit etre nulle par
construction, il rend +0.0147 R avec une erreur-type de 0.0181, soit un t de
0.81 : indiscernable de zero. Le taux de reussite mesure, 17.8 a 19.7%,
colle au taux theorique de 17.1 a 17.3% qu'impose une cible a 4.8 R. Le
moteur ne fabrique donc pas de gain.

C'est la precaution qui manquait au moteur d'Order Block, lequel rendait
+0.4565 R sur le meme test parce qu'il anticipait.

CE QUI EST COMPARE
  k            largeur de la bande, en ecarts-types
  poids        "tickvol" ou "plat". Sur l'or au comptant il n'existe pas de
               vrai volume : MT5 publie un nombre de TICKS, propre au
               courtier. Si les deux donnent le meme resultat, la
               ponderation ne sert a rien et le mot VWAP est usurpe.
  sens         "retour" contre "cassure", qui ne peuvent pas etre vrais
               tous les deux
  stop_sigma   distance du stop, en ecarts-types

Et toujours la meme mesure : l'ecart aux temoins, ou les bougies M1 ont ete
melangees a l'interieur de chaque heure, puis le decoupage en deux moities.

    python test_vwap.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from quant.mt5_loader import load_mt5_csv, resample
from quant.vwap import VwapConfig, backtest_vwap
from quant.surrogate import shuffle_bars

N_TEMOINS = 4
GRAINE = 20261013
RISQUE_EUR = 4.50          # 1% de 450 euros
JOURS_MOIS = 21

CONFIGS = [
    ("retour k=1.5",            dict(sens="retour", k=1.5)),
    ("retour k=2.0",            dict(sens="retour", k=2.0)),
    ("retour k=2.5",            dict(sens="retour", k=2.5)),
    ("retour k=3.0",            dict(sens="retour", k=3.0)),
    ("retour k=2.0, poids plat", dict(sens="retour", k=2.0, poids="plat")),
    ("retour k=2.0, stop 1.0s", dict(sens="retour", k=2.0, stop_sigma=1.0)),
    ("cassure k=2.0, cible 2R", dict(sens="cassure", k=2.0, cible="r",
                                     cible_r=2.0)),
    ("retour k=2.0, cible 2R",  dict(sens="retour", k=2.0, cible="r",
                                     cible_r=2.0)),
]


def mesure(m1: pd.DataFrame, opts: dict) -> dict:
    tr, st = backtest_vwap(m1, VwapConfig(**opts))
    if tr.empty:
        return {}
    return {"n": len(tr), "jour": st["trades_par_jour"],
            "wr": st["winrate"] * 100, "R": tr.R.mean(),
            "sd": tr.R.std(ddof=1), "stop": tr.stop_dist.median(),
            "cr": tr.cible_r.median()}


def bloc(m1: pd.DataFrame, titre: str, graine: int) -> dict:
    jours = m1.index.normalize().nunique()
    print(f"\n{titre}  ({jours} jours)", flush=True)
    print("  configuration                n   freq    WR  WRseuil  stop"
          "  cible    R    EUR/mois  temoin   ecart     t")
    print("  " + "-" * 104, flush=True)
    tem = [shuffle_bars(m1, seed=graine + k) for k in range(N_TEMOINS)]
    out = {}
    for nom, opts in CONFIGS:
        r = mesure(m1, opts)
        if not r or r["n"] < 30:
            print(f"  {nom:<26} {r.get('n', 0):>5}  (trop peu de trades)",
                  flush=True)
            continue
        rs = [x for x in (mesure(t, opts) for t in tem) if x]
        rt = float(np.mean([x["R"] for x in rs])) if rs else np.nan
        ec = r["R"] - rt
        t = ec / (r["sd"] / np.sqrt(r["n"])) if r["sd"] > 0 else np.nan
        eur = r["R"] * RISQUE_EUR * r["jour"] * JOURS_MOIS
        seuil = 100.0 / (1.0 + r["cr"])      # winrate d'equilibre
        out[nom] = (ec, r["jour"], r["R"], r["wr"], eur)
        print(f"  {nom:<26} {r['n']:>5} {r['jour']:>5.2f} {r['wr']:5.1f}% "
              f"{seuil:6.1f}% {r['stop']:5.2f} {r['cr']:5.2f}R "
              f"{r['R']:+.3f} {eur:>+8.0f}  {rt:+.3f}  {ec:+.3f}  {t:+5.2f}",
              flush=True)
    return out


def main(chemin: str) -> None:
    t0 = time.time()
    m1 = resample(load_mt5_csv(chemin), "1min")
    print(f"{len(m1):,} bougies M1 | VWAP de seance, bandes a k ecarts-types")
    print("MOTEUR VALIDE : rend +0.015 R (t 0.81) sur marche aleatoire sans")
    print("derive, ou la reponse doit etre zero. Il ne fabrique pas de gain.")
    print("'WRseuil' = taux de reussite d'equilibre, soit 1/(1+cible). Un")
    print("winrate superieur a ce seuil gagne, un winrate inferieur perd,")
    print("quelle que soit son allure.")

    tout = bloc(m1, "ECHANTILLON COMPLET", GRAINE)
    coupe = m1.index[len(m1) // 2]
    a = bloc(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100)
    b = bloc(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200)

    print("\n\nVERDICT")
    print("  configuration                 WR      R    EUR/mois   ecart"
          "      1re      2e     verdict")
    print("  " + "-" * 98)
    for nom, _ in CONFIGS:
        if nom not in tout:
            continue
        ec, fr, R, wr, eur = tout[nom]
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
        print(f"  {nom:<26} {wr:5.1f}% {R:+.3f} {eur:>+8.0f}  {ec:+.3f}"
              f"   {va:+.3f}  {vb:+.3f}   {v}")

    print("\n  EUR/mois suppose 1% de risque sur 450 EUR, soit 4.50 EUR par")
    print("  trade. C'est le resultat BRUT du backtest : l'ecart au temoin")
    print("  dit ce qu'il vaut. Un systeme qui gagne autant sur des bougies")
    print("  melangees gagne par geometrie, pas par lecture du marche.")
    print(f"\n  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
