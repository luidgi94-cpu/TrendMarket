"""Les sept criteres de qualification demandes, mesures un par un.

CE QUI EST DEMANDE
  1. l'Order Block genere une imbalance
  2. il n'est pas mitige
  3. pendant l'impulsion, aucune meche ne revient dans la zone
  4. pas de liquidite juste au-dela des plus bas ou des plus hauts
  5. la zone est en discount pour un achat, en premium pour une vente,
     le fibonacci etant trace du plus bas au plus haut du mouvement pour un
     achat et du plus haut au plus bas pour une vente
  6. entree au minimum au contact de la zone
  7. stop sous la zone avec tampon, cible 100 pips

TROIS PRECISIONS AVANT LES CHIFFRES
Le critere 2 n'a pas besoin d'etre ajoute : le moteur prend la PREMIERE
touche de chaque zone, une zone deja retouchee n'est donc jamais tradee une
seconde fois. C'est la definition de la non-mitigation, et elle est deja en
place depuis le debut. Le critere 3 en est la version forte, etendue a toute
l'impulsion, et lui est bien un ajout.

Le critere 7 fixe la cible en pips. Sur XAUUSD la convention du courtier est
1 pip = 0.10 USD l'once, donc 100 pips = 10.00 USD. Le stop median mesure
3.5 a 4.5 USD : la cible vaut donc environ 2.5 R, et non 17 R. C'est un
changement de geometrie complet par rapport a la cible proportionnelle de
3 R utilisee dans tous les tests precedents, et il est mesure a part.

Le critere 4 est pris dans le sens demande : on REFUSE la zone si un niveau
evident se trouve juste au-dela du stop. L'enseignement SMC dit souvent
l'inverse, que cette liquidite alimente le mouvement. Les deux lectures sont
mesurees, parce que l'une des deux est fausse et que seule la mesure peut
dire laquelle.

LE SEUL CHIFFRE QUI COMPTE
L'ecart au temoin. Les temoins melangent les bougies M1 a l'interieur de
chaque heure : la volatilite horaire et la forme des bougies survivent,
l'enchainement disparait. Un systeme qui gagne autant sur des bougies
melangees gagne par geometrie, pas par lecture du marche. Et le decoupage en
deux moities decide : il a elimine les neuf familles d'hypotheses testees
jusqu'ici.

    python test_criteres_ob.py chemin/vers/XAUUSD_M1.csv
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

# Plafond journalier releve a 50 : a 14 trades par jour sans filtre, le
# plafond de 10 etait saturant et masquait l'effet des criteres sur la
# frequence. Il est identique pour toutes les lignes, donc comparable.
BASE = dict(min_trend_force=0.0, partial_r=1e9, max_trades_per_day=50)
NOUVELLE = dict(entry_mode="contact", tp_usd=10.0)
TOUS = dict(require_imbalance=True, clean_impulse=True,
            liq_rule="absente", fib_max=0.5)

CONFIGS = [
    # (etiquette, options)
    ("reference 3 R, confirmation", dict()),
    ("contact + TP 100 pips",       dict(**NOUVELLE)),
    ("+ imbalance",                 dict(**NOUVELLE, require_imbalance=True)),
    ("+ impulsion propre",          dict(**NOUVELLE, require_imbalance=True,
                                         clean_impulse=True)),
    ("+ liquidite absente",         dict(**NOUVELLE, require_imbalance=True,
                                         clean_impulse=True,
                                         liq_rule="absente")),
    ("+ discount / premium  = TOUS", dict(**NOUVELLE, **TOUS)),
    ("TOUS, liquidite EXIGEE",      dict(NOUVELLE, **{**TOUS, "liq_rule": "presente"})),
    ("TOUS, TP 50 pips",            dict(NOUVELLE, **TOUS, tp_usd=5.0)),
    ("TOUS, TP 200 pips",           dict(NOUVELLE, **TOUS, tp_usd=20.0)),
]


def mesure(m5: pd.DataFrame, m1: pd.DataFrame, opts: dict) -> dict:
    cfg = MTFConfig(**BASE, **opts)
    tr, st = backtest_mtf(m5, m1, cfg)
    if tr.empty:
        return {}
    return {"n": len(tr), "jour": st["trades_par_jour"],
            "wr": st["winrate"] * 100, "R": tr.R.mean(),
            "sd": tr.R.std(ddof=1), "stop": tr.stop_dist.median(),
            "tp": tr.tp_en_r.median()}


def bloc(m1: pd.DataFrame, titre: str, graine: int) -> dict:
    m5 = resample(m1, "5min")
    jours = m5.index.normalize().nunique()
    print(f"\n{titre}  ({jours} jours)", flush=True)
    print("  configuration                   n   freq    WR   stop   TP"
          "    R reel   temoin   ecart     t")
    print("  " + "-" * 88, flush=True)
    tem1 = [shuffle_bars(m1, seed=graine + k) for k in range(N_TEMOINS)]
    tem5 = [resample(t, "5min") for t in tem1]
    out = {}
    for nom, opts in CONFIGS:
        r = mesure(m5, m1, opts)
        if not r or r["n"] < 30:
            print(f"  {nom:<28} {r.get('n', 0):>5}  (trop peu de trades)",
                  flush=True)
            continue
        rs = [mesure(t5, t1, opts) for t5, t1 in zip(tem5, tem1)]
        rs = [x for x in rs if x]
        rt = float(np.mean([x["R"] for x in rs])) if rs else np.nan
        ec = r["R"] - rt
        t = ec / (r["sd"] / np.sqrt(r["n"])) if r["sd"] > 0 else np.nan
        out[nom] = (ec, r["jour"], r["n"])
        print(f"  {nom:<28} {r['n']:>5} {r['jour']:>5.2f} {r['wr']:5.1f}% "
              f"{r['stop']:5.2f} {r['tp']:4.2f}R  {r['R']:+.3f}  {rt:+.3f}  "
              f"{ec:+.3f}  {t:+5.2f}", flush=True)
    return out


def main(chemin: str) -> None:
    t0 = time.time()
    m1 = resample(load_mt5_csv(chemin), "1min")
    print(f"{len(m1):,} bougies M1 | 100 pips = 10.00 USD l'once")
    print("Criteres appliques A LA DETECTION : refuser une zone libere la")
    print("place pour la suivante, donc le deroulement change reellement.")
    print("'ecart' = R reel moins R des temoins. Lui seul mesure quelque")
    print("chose ; le R reel seul recompense la geometrie du systeme.")

    tout = bloc(m1, "ECHANTILLON COMPLET", GRAINE)
    coupe = m1.index[len(m1) // 2]
    a = bloc(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100)
    b = bloc(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200)

    print("\n\nSTABILITE  (le critere qui a elimine les neuf familles"
          " precedentes)")
    print("  configuration                complet    1re      2e     freq"
          "    verdict")
    print("  " + "-" * 82)
    for nom, _ in CONFIGS:
        if nom not in tout:
            continue
        nt, fr, _n = tout[nom]
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
        print(f"  {nom:<28} {nt:+.3f}   {va:+.3f}  {vb:+.3f}  {fr:5.2f}/j  {v}")

    print("\nTAILLE D'ECHANTILLON NECESSAIRE")
    print("  Pour distinguer un ecart reel d'un hasard, il faut environ")
    print("  n = (1.96 x ecart-type / ecart)^2 trades. Avec un ecart-type de")
    print("  1.2 R, un ecart de +0.10 R demande 553 trades, un ecart de")
    print("  +0.05 R en demande 2213. A 0.6 trade par jour, cela fait")
    print("  respectivement trois ans et douze ans de forward test.")
    print(f"\n  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
