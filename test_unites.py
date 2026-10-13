"""La methode marche-t-elle mieux sur une unite de temps plus grande ?

M5, M15, M30, H1, H4, sur les memes 436 jours de donnees du courtier. La
detection des zones change d'unite, la GESTION reste en M1 dans tous les
cas : c'est la seule facon de savoir si le stop a ete touche avant la cible
a l'interieur d'une grande bougie. Un backtest H4 qui lit le resultat sur
des bougies H4 ne peut pas le savoir, et se trompe toujours du bon cote.

CE QUI EST MIS A L'ECHELLE
L'impulsion reste de cinq bougies et la zone vit quarante-huit bougies dans
toutes les unites : ce sont des durees relatives, pas absolues. L'ATR couvre
vingt-quatre heures, donc 288 bougies en M5 et 14 en H4. La duree maximale
de detention suit l'unite, quatre heures en M5 et huit jours en H4 : garder
une position H4 quatre heures n'aurait aucun sens.

DEUX LIGNES PAR UNITE
  sans filtre        toutes les zones, entree au contact, cible 100 pips
  tous les criteres  imbalance, impulsion propre, pas de liquidite au-dela,
                     discount pour un achat et premium pour une vente

Et toujours la meme mesure : l'ecart aux temoins, ou les bougies M1 ont ete
melangees a l'interieur de chaque heure. La rentabilite affichee en euros
suppose 1% de risque sur 450 euros, soit 4.50 euros par trade.

    python test_unites.py chemin/vers/XAUUSD_M1.csv
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
RISQUE_EUR = 4.50          # 1% de 450 euros
JOURS_MOIS = 21

UNITES = [("M5", "5min", 5), ("M15", "15min", 15), ("M30", "30min", 30),
          ("H1", "1h", 60), ("H4", "4h", 240)]

TOUS = dict(require_imbalance=True, clean_impulse=True,
            liq_rule="absente", fib_max=0.5)
VARIANTES = [("sans filtre", dict()), ("tous les criteres", TOUS)]


def config(tf_min: int, opts: dict) -> MTFConfig:
    return MTFConfig(
        min_trend_force=0.0, partial_r=1e9, max_trades_per_day=50,
        entry_mode="contact", tp_usd=10.0,
        tf_minutes=tf_min,
        atr_bars=max(14, 1440 // tf_min),      # vingt-quatre heures
        max_hold_min=48 * tf_min,              # duree de vie de la zone
        **opts)


def mesure(m1: pd.DataFrame, regle: str, tf_min: int, opts: dict) -> dict:
    det = resample(m1, regle)
    tr, st = backtest_mtf(det, m1, config(tf_min, opts))
    if tr.empty:
        return {}
    return {"n": len(tr), "jour": st["trades_par_jour"],
            "wr": st["winrate"] * 100, "R": tr.R.mean(),
            "sd": tr.R.std(ddof=1), "stop": tr.stop_dist.median(),
            "tp": tr.tp_en_r.median()}


def bloc(m1: pd.DataFrame, titre: str, graine: int) -> dict:
    jours = m1.index.normalize().nunique()
    print(f"\n{titre}  ({jours} jours)", flush=True)
    print("  unite  configuration          n   freq    WR   stop    TP"
          "    R      EUR/mois   temoin   ecart     t")
    print("  " + "-" * 94, flush=True)
    tem = [shuffle_bars(m1, seed=graine + k) for k in range(N_TEMOINS)]
    out = {}
    for nom_u, regle, tf_min in UNITES:
        for nom_v, opts in VARIANTES:
            r = mesure(m1, regle, tf_min, opts)
            if not r or r["n"] < 30:
                print(f"  {nom_u:<6} {nom_v:<20} {r.get('n', 0):>5}"
                      f"  (trop peu de trades pour conclure)", flush=True)
                continue
            rs = [x for x in (mesure(t, regle, tf_min, opts) for t in tem) if x]
            rt = float(np.mean([x["R"] for x in rs])) if rs else np.nan
            ec = r["R"] - rt
            t = ec / (r["sd"] / np.sqrt(r["n"])) if r["sd"] > 0 else np.nan
            eur = r["R"] * RISQUE_EUR * r["jour"] * JOURS_MOIS
            out[(nom_u, nom_v)] = (ec, r["jour"], r["R"], r["wr"], eur)
            print(f"  {nom_u:<6} {nom_v:<20} {r['n']:>5} {r['jour']:>5.2f} "
                  f"{r['wr']:5.1f}% {r['stop']:5.2f} {r['tp']:5.2f}R "
                  f"{r['R']:+.3f} {eur:>+9.0f}   {rt:+.3f}  {ec:+.3f}  "
                  f"{t:+5.2f}", flush=True)
    return out


def main(chemin: str) -> None:
    t0 = time.time()
    m1 = resample(load_mt5_csv(chemin), "1min")
    print(f"{len(m1):,} bougies M1 | detection sur l'unite indiquee,")
    print("GESTION EN M1 DANS TOUS LES CAS, pour que l'ordre entre le stop")
    print("et la cible soit connu a l'interieur des grandes bougies.")
    print("EUR/mois : 1% de risque sur 450 EUR, soit 4.50 EUR par trade,")
    print("et 21 jours de bourse. C'est la rentabilite BRUTE du backtest,")
    print("pas une prevision : l'ecart au temoin dit ce qu'elle vaut.")

    tout = bloc(m1, "ECHANTILLON COMPLET", GRAINE)
    coupe = m1.index[len(m1) // 2]
    a = bloc(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100)
    b = bloc(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200)

    print("\n\nRECAPITULATIF : RENTABILITE, WINRATE, ET CE QU'ELLE VAUT")
    print("  unite  configuration            WR      R     EUR/mois"
          "   ecart temoin        verdict")
    print("  " + "-" * 88)
    for cle in tout:
        ec, fr, R, wr, eur = tout[cle]
        va = a.get(cle, (np.nan,))[0]
        vb = b.get(cle, (np.nan,))[0]
        if np.isnan(va) or np.isnan(vb):
            v = "indeterminable"
        elif va > 0 and vb > 0:
            v = "POSITIF DES DEUX COTES"
        elif va * vb < 0:
            v = "s'inverse"
        else:
            v = "negatif des deux cotes"
        print(f"  {cle[0]:<6} {cle[1]:<22} {wr:5.1f}% {R:+.3f} {eur:>+9.0f}"
              f"     {ec:+.3f}        {v}")

    print("\n  Le winrate et la rentabilite se lisent dans les deux premieres")
    print("  colonnes. L'ecart au temoin dit si cette rentabilite vient d'une")
    print("  lecture du marche ou seulement de la forme du pari : un systeme")
    print("  qui gagne autant sur des bougies melangees ne gagnera pas en")
    print("  reel, ou les bougies ne sont pas melangees mais ne se repetent")
    print("  pas non plus.")
    print(f"\n  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
