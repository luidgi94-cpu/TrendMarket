"""Les ecarts observes sont-ils distinguables de zero ?

L'etude des attributs a fait ressortir deux criteres qui gardent le bon
signe sur les deux moities : l'imbalance et l'impulsion propre. C'est une
premiere dans ce projet. Reste a savoir si ces ecarts sont mesurables, car
les groupes concernes sont petits : 404 et 205 trades sur 6 262.

Un ecart de +0.10 R sur 200 trades, avec un ecart-type de 1.4 R, a une
erreur-type de 0.14 R. Il est donc indistinguable de zero. Ce script calcule
l'erreur-type et le t de Welch de chaque ecart, groupe par groupe, bloc par
bloc, pour que la question soit tranchee par un chiffre et non par
l'apparence d'un tableau.

Il affiche aussi, pour chaque critere, le nombre de trades qu'il faudrait
pour que l'ecart observe devienne mesurable, et combien d'annees de forward
test cela represente a la frequence constatee.

    python test_criteres_signif.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quant.mt5_loader import load_mt5_csv, resample
from quant.scalp_mtf import MTFConfig, backtest_mtf

BINAIRES = {
    "ob_plus":  "imbalance generee",
    "a_propre": "impulsion propre",
    "a_liq":    "liquidite juste au-dela",
}


def welch(g1: pd.Series, g0: pd.Series) -> tuple[float, float, float]:
    """Ecart des moyennes, son erreur-type, et le t correspondant."""
    ec = g1.mean() - g0.mean()
    se = float(np.sqrt(g1.var(ddof=1) / len(g1) + g0.var(ddof=1) / len(g0)))
    return ec, se, (ec / se if se > 0 else np.nan)


def bloc(m1: pd.DataFrame, titre: str) -> None:
    cfg = MTFConfig(min_trend_force=0.0, partial_r=1e9, max_trades_per_day=50,
                    entry_mode="contact", tp_usd=10.0)
    tr, st = backtest_mtf(resample(m1, "5min"), m1, cfg)
    jours = st["jours"]
    print(f"\n{titre}  ({len(tr)} trades, {jours} jours, "
          f"{len(tr) / jours:.2f}/j, ecart-type {tr.R.std(ddof=1):.2f} R)")
    print("  critere                        n       R     ecart      se"
          "      t      n requis   annees")
    print("  " + "-" * 84)
    for col, nom in BINAIRES.items():
        v = tr[[col, "R"]].dropna()
        g0 = v.loc[v[col] == 0, "R"]
        g1 = v.loc[v[col] == 1, "R"]
        if len(g0) < 30 or len(g1) < 30:
            print(f"  {nom:<28} groupes {len(g0)} / {len(g1)} : trop petits")
            continue
        ec, se, t = welch(g1, g0)
        # Nombre de trades du groupe retenu pour que cet ecart soit mesurable
        # a 95%, et duree correspondante a la frequence de ce groupe.
        if abs(ec) > 0:
            n_req = (1.96 * g1.std(ddof=1) / abs(ec)) ** 2
            freq = len(g1) / jours
            annees = n_req / freq / 252 if freq > 0 else np.inf
        else:
            n_req, annees = np.inf, np.inf
        print(f"  {nom:<28} {len(g1):>5} {g1.mean():+.3f}  {ec:+.3f}  "
              f"{se:.3f}  {t:+5.2f}   {n_req:>8.0f}  {annees:>7.1f}")

    # Position dans le mouvement : quintile extreme contre quintile extreme
    v = tr[["a_fib", "R"]].dropna()
    q = pd.qcut(v.a_fib, 5, labels=False, duplicates="drop")
    g1 = v.loc[q == q.max(), "R"]
    g0 = v.loc[q == 0, "R"]
    ec, se, t = welch(g1, g0)
    print(f"  {'position Q5 contre Q1':<28} {len(g1):>5} {g1.mean():+.3f}  "
          f"{ec:+.3f}  {se:.3f}  {t:+5.2f}")
    print("    Q5 est le quintile le plus PREMIUM pour un achat, donc un t")
    print("    positif contredit le critere discount demande.")


def main(chemin: str) -> None:
    m1 = resample(load_mt5_csv(chemin), "1min")
    print(f"{len(m1):,} bougies M1 | entree au contact, cible 100 pips")
    print("Aucun critere ne filtre. 'ecart' compare les trades qui respectent")
    print("le critere a ceux qui ne le respectent pas, au sein du MEME")
    print("echantillon : les temoins ne sont donc pas necessaires ici, la")
    print("geometrie du systeme etant identique dans les deux groupes.")
    print("'n requis' = trades du groupe necessaires pour que l'ecart observe")
    print("soit distinguable de zero a 95%, et 'annees' la duree de forward")
    print("test correspondante, a 252 jours de bourse par an.")

    bloc(m1, "ECHANTILLON COMPLET")
    coupe = m1.index[len(m1) // 2]
    bloc(m1[m1.index < coupe], "PREMIERE MOITIE")
    bloc(m1[m1.index >= coupe], "SECONDE MOITIE")

    print("\nLECTURE")
    print("  Un t inferieur a 2 en valeur absolue ne permet pas de conclure.")
    print("  Et il faut etre plus exigeant que 2 ici : une quarantaine de")
    print("  configurations ont deja ete essayees dans ce projet, si bien")
    print("  qu'un t de 2 isole ne vaut plus 5% mais environ 10 a 15% apres")
    print("  correction pour la multiplicite des essais.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
