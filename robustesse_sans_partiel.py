"""Le retrait de la prise partielle revele-t-il un edge, ou un artefact ?

Le balayage precedent a produit, au seuil 0.8 et sans prise partielle, un
ecart au temoin de +0.140 R avec t = 2.99. Pris isolement ce chiffre
parait concluant. Il ne l'est pas, pour trois raisons :

  - il est issu de 36 configurations testees (9 seuils x 4 variantes) ;
  - le seuil 0.8 n'a pas ete choisi a l'avance, il a ete retenu PARCE
    QU'il affichait le meilleur ecart ;
  - le filtre de largeur de stop y est applique a posteriori.

Ce script applique les deux contre-mesures qui tranchent.

TEST 1 - L'effet est-il une propriete large ou une cellule chanceuse ?
Si retirer le partiel revele du signal, l'effet doit se voir a TOUS les
seuils, pas au seul qui a ete pioche. On mesure donc l'ecart au temoin,
avec et sans partiel, sur les neuf seuils. Ce qui compte n'est pas qu'un
seuil sorte du lot, c'est que la colonne entiere se deplace.

TEST 2 - Hors echantillon temporel.
On choisit le seuil sur la PREMIERE moitie des donnees, puis on le mesure
sur la SECONDE, jamais vue. C'est la seule facon de savoir si le reglage
capture une propriete du marche ou la forme particuliere de cet
echantillon. Un edge qui ne survit pas a ce decoupage n'existe pas.

    python robustesse_sans_partiel.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quant.mt5_loader import load_mt5_csv, resample
from quant.surrogate import shuffle_bars
from balayage_force import mesure, SEUILS

N_TEMOINS = 5
GRAINE = 20260916


def ecart(m5, m1, temoins, seuil: float, partiel: bool) -> tuple:
    """Retourne (ecart au temoin, t de Student, nb de trades, R, WR)."""
    r = mesure(m5, m1, seuil, partiel=partiel)
    if r["trades"] < 2:
        return np.nan, np.nan, r["trades"], np.nan, np.nan
    rs = [mesure(t5, t1, seuil, partiel=partiel) for t5, t1 in temoins]
    rs = [x for x in rs if x["trades"] > 0]
    if not rs:
        return np.nan, np.nan, r["trades"], r["R_moyen"], r["winrate"]
    rt = float(np.mean([x["R_moyen"] for x in rs]))
    ec = r["R_moyen"] - rt
    err = r["R_ecart"] / np.sqrt(r["trades"])
    return ec, (ec / err if err > 0 else np.nan), r["trades"], r["R_moyen"], r["winrate"]


def construire_temoins(m1: pd.DataFrame, n: int, graine: int) -> list:
    out = []
    for k in range(n):
        t1 = shuffle_bars(m1, seed=graine + k)
        out.append((resample(t1, "5min"), t1))
    return out


def main(chemin: str) -> None:
    brut = load_mt5_csv(chemin)
    m1 = resample(brut, "1min")
    m5 = resample(brut, "5min")
    print(f"{len(m1):,} bougies M1 | {m5.index.normalize().nunique()} jours\n")

    print(f"Construction de {N_TEMOINS} temoins (echantillon complet)...")
    tem = construire_temoins(m1, N_TEMOINS, GRAINE)
    print("fait.\n")

    # ---------------------------------------------------------- TEST 1 ----
    print("TEST 1 - l'effet du retrait du partiel, seuil par seuil")
    print("  Si l'effet est reel, la colonne 'sans' doit dominer la colonne")
    print("  'avec' PARTOUT, pas seulement au seuil qui a ete pioche.\n")
    print("  seuil |    avec partiel     |    SANS partiel     |  gain")
    print("        |  ecart      t    WR |  ecart      t    WR |")
    print("  " + "-" * 62)
    lignes = []
    for s in SEUILS:
        ea, ta, na_, ra, wa = ecart(m5, m1, tem, s, True)
        es, ts, ns, rs_, ws = ecart(m5, m1, tem, s, False)
        lignes.append({"seuil": s, "ecart_avec": ea, "t_avec": ta,
                       "ecart_sans": es, "t_sans": ts, "n": ns})
        print(f"  {s:<5} | {ea:+.3f}  {ta:5.2f}  {wa:4.1f}% | "
              f"{es:+.3f}  {ts:5.2f}  {ws:4.1f}% | {es - ea:+.3f}")

    df = pd.DataFrame(lignes)
    gains = (df.ecart_sans - df.ecart_avec).dropna()
    pos = int((gains > 0).sum())
    print(f"\n  Le retrait du partiel ameliore l'ecart sur {pos}/{len(gains)} seuils.")
    print(f"  Gain median {gains.median():+.3f} R, minimum {gains.min():+.3f}, "
          f"maximum {gains.max():+.3f}.")
    if pos == len(gains):
        print("  -> effet SYSTEMATIQUE. Ce n'est pas une cellule chanceuse.")
    elif pos >= len(gains) * 0.7:
        print("  -> effet MAJORITAIRE mais pas universel. A confirmer.")
    else:
        print("  -> effet NON SYSTEMATIQUE : le resultat du seuil 0.8 etait")
        print("     tres probablement un artefact de selection.")

    # ---------------------------------------------------------- TEST 2 ----
    # Le seuil est choisi sur la premiere moitie, mesure sur la seconde.
    # Les temoins sont reconstruits sur chaque moitie : melanger le tout
    # puis decouper ferait fuiter de l'information d'une moitie a l'autre.
    coupe = m5.index[len(m5) // 2]
    print(f"\n\nTEST 2 - hors echantillon temporel (coupure {coupe:%Y-%m-%d})")
    m1a, m1b = m1[m1.index < coupe], m1[m1.index >= coupe]
    m5a, m5b = m5[m5.index < coupe], m5[m5.index >= coupe]
    print(f"  apprentissage {m5a.index.normalize().nunique()} jours | "
          f"validation {m5b.index.normalize().nunique()} jours")

    print("  Construction des temoins par moitie...")
    tem_a = construire_temoins(m1a, N_TEMOINS, GRAINE + 100)
    tem_b = construire_temoins(m1b, N_TEMOINS, GRAINE + 200)
    print("  fait.\n")

    print("  APPRENTISSAGE (on choisit le seuil ici, sans partiel)")
    best, best_ec = None, -np.inf
    for s in SEUILS:
        ec, t, n, r, w = ecart(m5a, m1a, tem_a, s, False)
        marque = ""
        if not np.isnan(ec) and ec > best_ec and n >= 200:
            best, best_ec, marque = s, ec, "  <-"
        print(f"    seuil {s:<5} ecart {ec:+.3f}  t {t:5.2f}  n {n:>5}{marque}")

    if best is None:
        print("\n  Aucun seuil ne reunit assez de trades. Test non concluant.")
        return

    print(f"\n  Seuil retenu sur l'apprentissage : {best}")
    print("\n  VALIDATION (seconde moitie, jamais vue)")
    ec, t, n, r, w = ecart(m5b, m1b, tem_b, best, False)
    print(f"    seuil {best} : {n} trades  WR {w:4.1f}%  R {r:+.3f}  "
          f"ecart {ec:+.3f}  t {t:5.2f}")

    print("\n  VERDICT")
    if np.isnan(t):
        print("    Indeterminable : pas assez de trades en validation.")
    elif t >= 2.0:
        print("    L'edge SURVIT au hors echantillon. C'est le seul resultat")
        print("    de ce projet qui autorise a envisager un capital reel, et")
        print("    il reste a confirmer en forward test, sur du slippage reel.")
    elif t > 0:
        print("    L'ecart reste POSITIF mais sous le seuil de conclusion.")
        print("    Insuffisant pour engager du capital : un edge qu'on ne")
        print("    sait pas distinguer du bruit se trade comme du bruit.")
    else:
        print("    L'edge DISPARAIT hors echantillon. Le reglage capturait la")
        print("    forme de la premiere moitie, pas une propriete du marche.")
        print("    Conclusion : pas d'edge exploitable. Ne pas trader.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
