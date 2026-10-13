"""D'ou viennent les 796 euros par mois du tableau sans filtre ?

LE PROBLEME
La ligne M5 sans filtre affiche 50.5% de reussite, +0.586 R par trade et
796 euros par mois sur un compte de 450. C'est joli, et c'est precisement
pour cela qu'il faut l'ouvrir. Les temoins, ou les bougies M1 ont ete
melangees, affichent +0.605 R, soit MIEUX que le reel. Un systeme qui gagne
autant sur des bougies dont l'ordre a ete detruit ne gagne pas grace a sa
lecture du marche. Reste a savoir grace a quoi.

L'HYPOTHESE A TESTER
Permuter des bougies ne change pas la SOMME de leurs variations : l'or a
monte de la meme quantite dans le temoin et dans le reel. Si le gain vient
de cette derive et non de la detection des zones, alors il doit se trouver
presque entierement du cote ACHAT, se concentrer sur les mois ou l'or a
monte, et disparaitre les autres mois. C'est falsifiable, et c'est ce que
mesure ce script.

CE QU'IL AFFICHE
  la derive de l'or sur la periode, en dollars et en ATR
  le R moyen par sens, achat contre vente
  le R moyen mois par mois, reel et temoin cote a cote
  la pire serie de pertes, en R et en euros, qui est la seule chose qui
  decide si un compte de 450 euros survit a ce systeme

    python test_dou_vient_le_gain.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quant.mt5_loader import load_mt5_csv, resample
from quant.scalp_mtf import MTFConfig, backtest_mtf
from quant.surrogate import shuffle_bars

RISQUE_EUR = 4.50          # 1% de 450 euros
CFG = dict(min_trend_force=0.0, partial_r=1e9, max_trades_per_day=50,
           entry_mode="contact", tp_usd=10.0)


def run(m1: pd.DataFrame) -> pd.DataFrame:
    tr, _ = backtest_mtf(resample(m1, "5min"), m1, MTFConfig(**CFG))
    return tr


def pire_serie(r: pd.Series) -> tuple[float, int]:
    """Pire recul cumule, en R, et sa duree en nombre de trades."""
    cum = r.cumsum().to_numpy()
    sommet = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    recul = cum - sommet
    i = int(np.argmin(recul))
    # duree : depuis le dernier sommet avant i
    j = i
    while j > 0 and cum[j - 1] >= sommet[i] - 1e-9:
        break
    debut = int(np.argmax(cum[:i + 1])) if i > 0 else 0
    return float(recul[i]), i - debut + 1


def main(chemin: str) -> None:
    m1 = resample(load_mt5_csv(chemin), "1min")
    px = m1.close
    print(f"{len(m1):,} bougies M1, du {px.index[0]:%d/%m/%Y} au "
          f"{px.index[-1]:%d/%m/%Y}")
    print(f"\nLA DERIVE DE L'OR SUR LA PERIODE")
    print(f"  prix au depart   {px.iloc[0]:>10.2f} USD l'once")
    print(f"  prix a l'arrivee {px.iloc[-1]:>10.2f}")
    print(f"  variation        {px.iloc[-1] - px.iloc[0]:>+10.2f} USD, soit "
          f"{100 * (px.iloc[-1] / px.iloc[0] - 1):+.1f}%")
    print("  Une permutation des bougies ne change pas cette somme : le")
    print("  temoin a connu exactement la meme derive que le reel.")

    reel = run(m1)
    tem = run(shuffle_bars(m1, seed=20261013))

    print(f"\n\nLE GAIN PAR SENS  ({len(reel)} trades)")
    print("  sens      n     part     WR      R moyen    contribution")
    print("  " + "-" * 60)
    for nom, g in reel.groupby("direction"):
        part = len(g) / len(reel)
        print(f"  {nom:<8} {len(g):>5} {100 * part:>6.1f}% "
              f"{100 * (g.pnl > 0).mean():>6.1f}% {g.R.mean():>+10.3f} "
              f"{g.R.mean() * part:>+13.3f}")
    print(f"  {'ensemble':<8} {len(reel):>5} {100.0:>6.1f}% "
          f"{100 * (reel.pnl > 0).mean():>6.1f}% {reel.R.mean():>+10.3f}")
    print("  'contribution' = R moyen du sens pondere par sa part. Si tout le")
    print("  gain est du cote achat, il s'agit de la derive de l'or, pas de")
    print("  la detection des zones.")

    print("\n\nMOIS PAR MOIS")
    print("  mois      or (USD)    n     WR     R reel   R temoin   ecart"
          "     EUR")
    print("  " + "-" * 72)
    mois_px = px.resample("MS").agg(["first", "last"])
    gr = reel.groupby(reel.entry_ts.dt.to_period("M"))
    gt = tem.groupby(tem.entry_ts.dt.to_period("M"))
    n_pos, n_tot = 0, 0
    for per, g in gr:
        var = np.nan
        cle = per.to_timestamp()
        if cle in mois_px.index:
            var = mois_px.loc[cle, "last"] - mois_px.loc[cle, "first"]
        rt = gt.get_group(per).R.mean() if per in gt.groups else np.nan
        ec = g.R.mean() - rt
        eur = g.R.sum() * RISQUE_EUR
        n_tot += 1
        if not np.isnan(ec) and ec > 0:
            n_pos += 1
        print(f"  {str(per):<9} {var:>+8.0f} {len(g):>5} "
              f"{100 * (g.pnl > 0).mean():>5.1f}% {g.R.mean():>+8.3f} "
              f"{rt:>+9.3f} {ec:>+8.3f} {eur:>+8.0f}")
    print(f"\n  Mois ou le reel fait mieux que le temoin : {n_pos} sur {n_tot}."
          f" Le hasard en donnerait la moitie.")

    print("\n\nCE QU'UN COMPTE DE 450 EUROS SUBIRAIT")
    for nom, tr in (("reel", reel), ("temoin", tem)):
        rec, duree = pire_serie(tr.R)
        jours = tr.date.nunique()
        print(f"  {nom:<8} pire recul {rec:>+8.2f} R  soit "
              f"{rec * RISQUE_EUR:>+8.0f} EUR  "
              f"({100 * rec * RISQUE_EUR / 450:+.0f}% du compte), "
              f"sur {duree} trades")
        print(f"           {len(tr) / jours:.1f} trades par jour a 1% de "
              f"risque = {len(tr) / jours:.0f}% du capital engage par jour")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
