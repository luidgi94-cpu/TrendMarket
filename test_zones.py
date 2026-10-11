"""Peut-on distinguer une bonne zone d'une mauvaise ?

C'est la question que pose l'utilisateur, et c'est la bonne. Les tests
precedents ont montre que SEPT facons d'entrer echouent de la meme
maniere, y compris l'entree sans aucune confirmation. Si le declencheur
n'est pas en cause, c'est la ZONE. Reste a savoir si les zones sont
distinguables entre elles, ou si elles se valent toutes.

LA DIFFERENCE AVEC CE QUI A DEJA ETE FAIT
L'imbalance, la fraicheur, la confluence et le volume ont ete testes UN
PAR UN, chacun dans son propre script. Ici, dix attributs sont enregistres
pour CHAQUE zone au moment de sa detection, puis examines ensemble sur le
meme echantillon de trades.

    a_imp     amplitude de l'impulsion, en ATR
    a_zone    epaisseur de la zone, en ATR
    a_corps   part du corps dans l'amplitude de la bougie d'Order Block
    a_jambe   distance parcourue depuis la zone avant le retour, en ATR
    a_age5    bougies M5 entre la bougie d'Order Block et l'impulsion
    a_bos     l'impulsion a-t-elle depasse le dernier pivot de meme sens
    a_vol     volume de la bougie d'Order Block, sur sa moyenne locale
    a_pos     position dans la fourchette du jour a la detection
    a_force   nettete de la tendance H1
    a_delai   minutes entre la creation de la zone et son premier contact

CE QUI EST MESURE
Pour chaque attribut, les trades sont ranges en cinq quintiles et l'on
regarde le R moyen de chacun. Deux choses comptent, et aucune seule ne
suffit :

  L'ECART entre le quintile haut et le quintile bas, compare a celui des
  TEMOINS. Un attribut qui separe sur des bougies melangees separe par
  geometrie, pas par information.

  La MONOTONIE, mesuree par la correlation de rang entre le numero du
  quintile et son R moyen. Un seul quintile extreme qui ressort est
  beaucoup moins convaincant qu'une progression reguliere.

Et la encore, le decoupage temporel decide : un attribut n'est retenu que
si sa separation tient sur les deux moities.

SI AUCUN ATTRIBUT NE SEPARE
Ce serait une reponse definitive, et utile : les zones seraient
indistinguables, et aucune selection, aussi fine soit-elle, ne pourrait
rendre la methode rentable. Mieux vaut le savoir que de chercher
indefiniment le bon critere.

    python test_zones.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quant.mt5_loader import load_mt5_csv, resample
from quant.scalp_mtf import MTFConfig, backtest_mtf
from quant.surrogate import shuffle_bars

N_TEMOINS = 4
GRAINE = 20261011
NQ = 5

ATTRS = {
    "a_imp":   "amplitude de l'impulsion",
    "a_zone":  "epaisseur de la zone",
    "a_corps": "corps / amplitude de l'OB",
    "a_jambe": "distance parcourue",
    "a_age5":  "age de l'OB a l'impulsion",
    "a_bos":   "cassure de structure",
    "a_vol":   "volume de la bougie OB",
    "a_pos":   "position dans le jour",
    "a_force": "nettete de la tendance",
    "a_delai": "delai avant le contact",
}


def trades(m1: pd.DataFrame) -> pd.DataFrame:
    cfg = MTFConfig(min_trend_force=0.0, partial_r=1e9)
    tr, _ = backtest_mtf(resample(m1, "5min"), m1, cfg)
    return tr


def quintiles(tr: pd.DataFrame, col: str) -> tuple:
    """R moyen par quintile, ecart haut-bas, et correlation de rang."""
    v = tr[[col, "R"]].dropna()
    if len(v) < 100 or v[col].nunique() < 3:
        return None, np.nan, np.nan
    try:
        q = pd.qcut(v[col], NQ, labels=False, duplicates="drop")
    except ValueError:
        return None, np.nan, np.nan
    moy = v.groupby(q).R.mean()
    if len(moy) < 3:
        return None, np.nan, np.nan
    ecart = moy.iloc[-1] - moy.iloc[0]
    # correlation de rang entre le numero du quintile et son R moyen
    rho = np.corrcoef(np.arange(len(moy)), moy.to_numpy())[0, 1]
    return moy, ecart, rho


def etude(m1: pd.DataFrame, titre: str, graine: int) -> dict:
    print(f"\n{titre}")
    reel = trades(m1)
    if reel.empty:
        print("  aucun trade"); return {}
    tems = [trades(shuffle_bars(m1, seed=graine + k)) for k in range(N_TEMOINS)]
    tems = [x for x in tems if not x.empty]

    print(f"  {len(reel)} trades reels, {len(tems)} temoins")
    print("  attribut                         Q1     Q3     Q5   ecart"
          "  temoin   net    rho")
    print("  " + "-" * 76)
    out = {}
    for col, nom in ATTRS.items():
        if col not in reel.columns:
            continue
        moy, ec, rho = quintiles(reel, col)
        if moy is None:
            print(f"  {nom:<30} (pas assez de valeurs distinctes)")
            continue
        ecs = [quintiles(t, col)[1] for t in tems]
        ecs = [x for x in ecs if not np.isnan(x)]
        et = float(np.mean(ecs)) if ecs else np.nan
        net = ec - et
        out[col] = (net, rho)
        q1 = moy.iloc[0]
        q3 = moy.iloc[len(moy) // 2]
        q5 = moy.iloc[-1]
        print(f"  {nom:<30} {q1:+.3f} {q3:+.3f} {q5:+.3f} {ec:+.3f} "
              f"{et:+.3f} {net:+.3f} {rho:+.2f}")
    return out


def main(chemin: str) -> None:
    m1 = resample(load_mt5_csv(chemin), "1min")
    print(f"{len(m1):,} bougies M1 | seuil de force 0, sans prise partielle")
    print("Dix attributs enregistres par zone, aucun ne filtre.")
    print("'ecart' = R du quintile haut moins R du quintile bas.")
    print("'net'   = cet ecart moins celui des temoins. Seul 'net' mesure")
    print("          de l'information ; 'ecart' seul recompense la geometrie.")
    print("'rho'   = regularite de la progression entre quintiles, de -1 a +1.")

    tout = etude(m1, "ECHANTILLON COMPLET (436 jours)", GRAINE)
    coupe = m1.index[len(m1) // 2]
    a = etude(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100)
    b = etude(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200)

    print("\n\nCE QUI TIENT SUR LES DEUX MOITIES")
    print("  attribut                      net complet   1re    2e    verdict")
    print("  " + "-" * 72)
    retenus = []
    for col, nom in ATTRS.items():
        if col not in tout:
            continue
        nt = tout[col][0]
        va = a.get(col, (np.nan,))[0]
        vb = b.get(col, (np.nan,))[0]
        if np.isnan(va) or np.isnan(vb):
            v = "indeterminable"
        elif abs(va) < 0.05 and abs(vb) < 0.05:
            v = "plat des deux cotes"
        elif va * vb > 0:
            v = "MEME SENS"
            retenus.append((col, nom, nt, va, vb))
        else:
            v = "s'inverse"
        print(f"  {nom:<30} {nt:+.3f}    {va:+.3f} {vb:+.3f}  {v}")

    print("\nVERDICT")
    if not retenus:
        print("  AUCUN attribut ne separe les zones de facon stable.")
        print("  Les zones sont donc indistinguables entre elles : aucune")
        print("  selection, aussi fine soit-elle, ne peut rendre la methode")
        print("  rentable. Le probleme n'est pas qu'on voit mal les bonnes")
        print("  zones, c'est qu'il n'y en a pas de bonnes a voir.")
    else:
        print("  Attributs dont la separation garde le meme sens sur les deux")
        print("  moities :")
        for col, nom, nt, va, vb in retenus:
            print(f"    {nom:<30} net {nt:+.3f}  ({va:+.3f} puis {vb:+.3f})")
        print("  A confirmer en construisant un filtre sur le ou les meilleurs,")
        print("  puis en le mesurant hors echantillon comme le reste.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
