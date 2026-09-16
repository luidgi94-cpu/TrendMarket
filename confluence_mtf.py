"""La confluence multi-echelle ajoute-t-elle quelque chose ?

HYPOTHESE testee, formulee par l'utilisateur : un Order Block de petite
unite de temps qui se trouve a l'interieur, ou au contact, d'un Order
Block d'une unite superieure aurait beaucoup plus de chances de produire
une reaction exploitable.

Elle a une logique mecanique defendable : une zone qui coincide sur deux
echelles a ete construite par des flux plus gros qu'une zone qui
n'apparait qu'en M5.

CE QUE L'ON MESURE, ET CE QUE L'ON REFUSE DE MESURER
On ne mesure pas "est-ce que ca reagit". Le prix reagit a tout extreme
local : les temoins le montrent, des bougies melangees au hasard
produisent le meme taux de reaction. On mesure si la reaction est plus
GRANDE que celle du hasard, en R, ce qui est la seule chose qui paie.

LA PRECAUTION QUI DECIDE DE LA VALIDITE DU TEST
La confluence est calculee A L'IDENTIQUE sur les donnees reelles et sur
chaque temoin. Comparer des trades reels selectionnes par confluence a
des trades temoins non selectionnes fabriquerait un edge de toutes
pieces : la selection elle-meme deplace la distribution. Chaque groupe
est donc confronte a SON propre temoin, groupe de la meme facon.

    python confluence_mtf.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quant.mt5_loader import load_mt5_csv, resample
from quant.scalp_mtf import MTFConfig, backtest_mtf, _atr
from quant.surrogate import shuffle_bars

N_TEMOINS = 4
GRAINE = 20260916
SEUIL = 0.8


def zones_htf(df: pd.DataFrame, cfg: MTFConfig) -> pd.DataFrame:
    """Order Blocks d'une unite superieure, avec leur fenetre de validite.

    Meme detection que le moteur principal : impulsion d'au moins
    impulse_atr x ATR sur impulse_bars bougies, puis derniere bougie de
    couleur opposee dans la fenetre situee entre impulse_bars et
    impulse_bars + 10 bougies en arriere, zone prise meches comprises.

    Une zone reste valide jusqu'a ce qu'une CLOTURE passe au-dela, ce qui
    est la meme regle d'invalidation que celle du moteur principal.
    """
    o, h = df.open.to_numpy(), df.high.to_numpy()
    l, c = df.low.to_numpy(), df.close.to_numpy()
    atr = _atr(df, 200)
    idx = df.index
    n = len(df)
    out = []
    for i in range(cfg.impulse_bars + 12, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        move = c[i] - o[i - cfg.impulse_bars + 1]
        if abs(move) < cfg.impulse_atr * atr[i]:
            continue
        d = 1 if move > 0 else -1
        ob = -1
        for j in range(i - cfg.impulse_bars, max(i - cfg.impulse_bars - 10, 1), -1):
            if (c[j] < o[j]) if d > 0 else (c[j] > o[j]):
                ob = j
                break
        if ob < 0:
            continue
        lo = min(o[ob], c[ob], l[ob])
        hi = max(o[ob], c[ob], h[ob])
        # validite : jusqu'a la premiere cloture au-dela de la zone
        fin = n - 1
        for k in range(i + 1, n):
            if (c[k] < lo) if d > 0 else (c[k] > hi):
                fin = k
                break
        out.append({"debut": idx[i], "fin": idx[fin], "lo": lo, "hi": hi, "dir": d})
    return pd.DataFrame(out)


def marque_confluence(trades: pd.DataFrame, prix: pd.Series,
                      zones: pd.DataFrame, tol: float) -> np.ndarray:
    """True si l'entree tombe dans une zone superieure de MEME sens.

    tol elargit la zone de quelques dollars de chaque cote : l'utilisateur
    parle d'un Order Block "a cote" de celui de l'unite superieure, pas
    seulement strictement a l'interieur.
    """
    if zones.empty or trades.empty:
        return np.zeros(len(trades), bool)
    ts = pd.DatetimeIndex(trades.entry_ts)
    px = prix.reindex(ts, method="ffill").to_numpy()
    d = np.where(trades.direction.to_numpy() == "long", 1, -1)
    zd = zones.debut.to_numpy()
    zf = zones.fin.to_numpy()
    zl = zones.lo.to_numpy() - tol
    zh = zones.hi.to_numpy() + tol
    zr = zones.dir.to_numpy()
    tsv = ts.to_numpy()
    res = np.zeros(len(trades), bool)
    for i in range(len(trades)):
        actif = (zd <= tsv[i]) & (zf >= tsv[i]) & (zr == d[i])
        if actif.any():
            res[i] = bool(((px[i] >= zl) & (px[i] <= zh) & actif).any())
    return res


def etude(m1: pd.DataFrame, cfg: MTFConfig, tol: float) -> dict:
    """Retourne le R moyen par groupe de confluence."""
    m5 = resample(m1, "5min")
    tr, _ = backtest_mtf(m5, m1, cfg)
    if tr.empty:
        return {}
    prix = m1.close
    h1 = zones_htf(resample(m1, "60min"), cfg)
    h4 = zones_htf(resample(m1, "240min"), cfg)
    c1 = marque_confluence(tr, prix, h1, tol)
    c4 = marque_confluence(tr, prix, h4, tol)
    return {"aucune": tr.R[~c1 & ~c4], "H1": tr.R[c1 & ~c4],
            "H4": tr.R[c4 & ~c1], "H1+H4": tr.R[c1 & c4],
            "tout": tr.R}


def main(chemin: str) -> None:
    brut = load_mt5_csv(chemin)
    m1 = resample(brut, "1min")
    cfg = MTFConfig(min_trend_force=SEUIL, partial_r=1e9)
    tol = 0.5 * float(np.nanmedian(_atr(resample(m1, "5min"), 200)))
    print(f"{len(m1):,} bougies M1 | {resample(m1,'5min').index.normalize().nunique()} jours")
    print(f"seuil de force {SEUIL}, sans prise partielle")
    print(f"tolerance de contact : {tol:.2f} USD de chaque cote de la zone\n")

    print("Donnees reelles...")
    reel = etude(m1, cfg, tol)

    print(f"Temoins ({N_TEMOINS}, confluence recalculee a l'identique)...")
    tem = {k: [] for k in reel}
    for k in range(N_TEMOINS):
        e = etude(shuffle_bars(m1, seed=GRAINE + k), cfg, tol)
        for g, v in e.items():
            if len(v):
                tem[g].append(v.mean())
    print("fait.\n")

    print("  groupe     n      WR      R reel   temoin    ecart      t")
    print("  " + "-" * 58)
    lignes = []
    for g in ("aucune", "H1", "H4", "H1+H4", "tout"):
        v = reel.get(g, pd.Series(dtype=float))
        if len(v) < 20:
            print(f"  {g:<9} {len(v):>4}   (trop peu de trades pour conclure)")
            continue
        rt = float(np.mean(tem[g])) if tem[g] else np.nan
        ec = v.mean() - rt
        t = ec / (v.std(ddof=1) / np.sqrt(len(v))) if v.std(ddof=1) > 0 else np.nan
        wr = (v > 0).mean() * 100
        sep = "  <- confluence" if g in ("H1", "H4", "H1+H4") else ""
        print(f"  {g:<9} {len(v):>4}  {wr:5.1f}%  {v.mean():+.3f}  {rt:+.3f}  "
              f"{ec:+.3f}  {t:+6.2f}{sep}")
        lignes.append((g, len(v), ec, t))

    print("\nLECTURE")
    print("  Pour que l'hypothese tienne, il ne suffit pas que les groupes en")
    print("  confluence affichent un R eleve : il faut que leur ECART au")
    print("  temoin depasse celui du groupe 'aucune'. Un R plus haut avec le")
    print("  meme ecart signifie seulement que la confluence selectionne des")
    print("  moments plus volatils, ce que le hasard exploite tout autant.")
    base = next((e for g, n, e, t in lignes if g == "aucune"), np.nan)
    if not np.isnan(base):
        mieux = [g for g, n, e, t in lignes
                 if g in ("H1", "H4", "H1+H4") and e > base]
        print(f"\n  Ecart du groupe sans confluence : {base:+.3f} R")
        if not mieux:
            print("  AUCUN groupe en confluence ne fait mieux.")
            print("  -> l'hypothese n'est pas soutenue par ces donnees.")
        else:
            print(f"  Groupes en confluence qui font mieux : {', '.join(mieux)}")
            print("  -> a soumettre au test hors echantillon avant toute")
            print("     conclusion. Un ecart superieur sur l'echantillon")
            print("     d'ajustement ne suffit pas.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
