"""Momentum + repli + ordre limite : y a-t-il quelque chose ?

LE PARI
Tendance haussiere, on attend un repli, on pose un ordre LIMITE dessous,
stop sous le niveau, cible proportionnelle au risque. Et l'inverse pour une
tendance baissiere. Il vient de deux mesures : sur les bandes de VWAP, seule
la cassure bat son temoin, de +0.104 R, donc l'or prolonge ses mouvements ;
mais la cassure perd quand meme, son temoin valant -0.198 R, parce qu'un
ordre stop remplit APRES le depart du mouvement. D'ou l'idee de garder la
direction et de changer l'ordre.

LE MOTEUR A ETE VERIFIE DANS LES DEUX SENS
Justesse : sur marche aleatoire sans derive, il rend +0.019 R avec un t de
0.87, soit zero, A CONDITION que le stop depasse environ cinq amplitudes de
bougie M1. En dessous il rendait -0.020 R, le stop sautant par simple
discretisation. La colonne stop/M1 ci-dessous surveille cela ligne par
ligne : une configuration sous 5 n'est pas interpretable.

Puissance : sur une serie ou une tendance a ete INJECTEE, il la retrouve.
A une intensite valant un cinquieme du bruit, il mesure +0.088 R avec un t
de 3.70, et -0.061 quand on le retourne. Un "rien trouve" de sa part a donc
un sens, ce qui n'etait garanti nulle part jusqu'ici dans ce projet.

CE QUI EST MESURE
L'ecart aux temoins, ou les bougies M1 ont ete melangees a l'interieur de
chaque heure : la volatilite horaire et la forme des bougies survivent,
l'enchainement disparait. Or l'enchainement est precisement ce qu'un pari de
momentum pretend exploiter. Puis la coupure en deux moities, qui a elimine
toutes les hypotheses de ce projet.

    python test_momentum.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from quant.momentum import MomConfig, backtest_momentum
from quant.surrogate import shuffle_bars

N_TEMOINS = 4
GRAINE = 20261124
RISQUE_EUR = 4.50          # 1% de 450 euros
JOURS_MOIS = 21

CONFIGS = [
    # Stops serres, du calibre que l'utilisateur trade reellement. Le biais
    # y est conservateur, environ -0.02 R, la bougie d'entree testant le
    # stop sans tester la cible : ces lignes sous-estiment plutot.
    ("repli 0.08, stop 0.06", dict(repli_vol=0.08, stop_vol=0.06)),
    ("repli 0.12, stop 0.06", dict(repli_vol=0.12, stop_vol=0.06)),
    ("repli 0.20, stop 0.06", dict(repli_vol=0.20, stop_vol=0.06)),
    # Stops au-dela du seuil de fiabilite du backtest.
    ("repli 0.12, stop 0.10", dict(repli_vol=0.12, stop_vol=0.10)),
    ("repli 0.20, stop 0.10", dict(repli_vol=0.20, stop_vol=0.10)),
    ("repli 0.20, stop 0.15", dict(repli_vol=0.20, stop_vol=0.15)),
    ("repli 0.30, stop 0.15", dict(repli_vol=0.30, stop_vol=0.15)),
    ("repli 0.20, cible 1R",  dict(repli_vol=0.20, stop_vol=0.10, cible_r=1.0)),
    ("repli 0.20, cible 3R",  dict(repli_vol=0.20, stop_vol=0.10, cible_r=3.0)),
    ("CONTRE repli 0.20",     dict(repli_vol=0.20, stop_vol=0.10,
                                   sens="contre")),
]


def charge(chemin: str) -> pd.DataFrame:
    df = pd.read_csv(chemin, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if "volume" in df.columns and "tickvol" not in df.columns:
        df = df.rename(columns={"volume": "tickvol"})
    return df[["open", "high", "low", "close"]
              + (["tickvol"] if "tickvol" in df.columns else [])]


def mesure(m1: pd.DataFrame, opts: dict) -> dict:
    tr, st = backtest_momentum(m1, MomConfig(**opts))
    if tr.empty:
        return {}
    return {"n": len(tr), "jour": st["trades_par_jour"],
            "wr": st["winrate"] * 100, "R": tr.R.mean(),
            "sd": tr.R.std(ddof=1), "stop": tr.stop_dist.median(),
            "cr": tr.cible_r.median()}


def bloc(m1: pd.DataFrame, titre: str, graine: int, rng_m1: float) -> dict:
    jours = m1.index.normalize().nunique()
    print(f"\n{titre}  ({jours} jours)", flush=True)
    print("  configuration            n   freq    WR  stop stop/M1 cible"
          "     R   EUR/mois  temoin   ecart     t")
    print("  " + "-" * 104, flush=True)
    tem = [shuffle_bars(m1, seed=graine + k) for k in range(N_TEMOINS)]
    out = {}
    for nom, opts in CONFIGS:
        r = mesure(m1, opts)
        if not r or r["n"] < 30:
            print(f"  {nom:<22} {r.get('n', 0):>5}  (trop peu de trades)",
                  flush=True)
            continue
        rs = [x for x in (mesure(t, opts) for t in tem) if x]
        rt = float(np.mean([x["R"] for x in rs])) if rs else np.nan
        ec = r["R"] - rt
        t = ec / (r["sd"] / np.sqrt(r["n"])) if r["sd"] > 0 else np.nan
        eur = r["R"] * RISQUE_EUR * r["jour"] * JOURS_MOIS
        ratio = r["stop"] / rng_m1
        out[nom] = (ec, r["jour"], r["R"], r["wr"], eur, ratio)
        marque = " " if ratio >= 5 else "!"   # stop petit devant la bougie
        print(f"  {nom:<22} {r['n']:>5} {r['jour']:>5.2f} {r['wr']:5.1f}% "
              f"{r['stop']:5.2f} {ratio:5.1f}{marque} {r['cr']:4.2f}R "
              f"{r['R']:+.3f} {eur:>+8.0f}  {rt:+.3f}  {ec:+.3f}  {t:+5.2f}",
              flush=True)
    return out


def main(chemin: str) -> None:
    t0 = time.time()
    m1 = charge(chemin)
    rng_m1 = float((m1.high - m1.low).median())
    print(f"{len(m1):,} bougies M1 | "
          f"{m1.index[0]:%d/%m/%Y} au {m1.index[-1]:%d/%m/%Y}")
    print(f"amplitude mediane d'une bougie M1 : {rng_m1:.3f} USD")
    print("Un '!' dans la colonne stop/M1 signale un stop petit devant la")
    print("bougie. Le biais y est CONSERVATEUR, environ -0.02 R : la bougie")
    print("d'entree teste le stop sans tester la cible. Ces lignes")
    print("sous-estiment donc plutot qu'elles ne flattent.")

    tout = bloc(m1, "ECHANTILLON COMPLET", GRAINE, rng_m1)
    coupe = m1.index[len(m1) // 2]
    a = bloc(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100, rng_m1)
    b = bloc(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200, rng_m1)

    print("\n\nVERDICT")
    print("  configuration             WR      R   EUR/mois  ecart"
          "      1re      2e     verdict")
    print("  " + "-" * 96)
    for nom, _ in CONFIGS:
        if nom not in tout:
            continue
        ec, fr, R, wr, eur, ratio = tout[nom]
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
        print(f"  {nom:<22} {wr:5.1f}% {R:+.3f} {eur:>+8.0f}  {ec:+.3f}"
              f"   {va:+.3f}  {vb:+.3f}   {v}")

    print("\n  Rappel : le moteur detecte une tendance valant un cinquieme du")
    print("  bruit avec un ecart de +0.088 R. Un ecart nettement plus petit")
    print("  que cela, meme positif, reste dans son bruit de mesure.")
    print(f"\n  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
