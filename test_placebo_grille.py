"""Le placebo tient-il hors de son reglage, ou est-il ajuste ?

LE RESULTAT A VERIFIER
La zone posee par simple geometrie, sans aucun Order Block, obtient un ecart
au temoin de +0.160 R avec un t de 3.44, positif sur les deux moities. Elle
fait mieux que la vraie detection d'Order Block, dont l'apport propre est de
-0.037 R.

LE DOUTE LEGITIME
Les deux reglages du placebo, 1.80 ATR de recul et 0.90 ATR d'epaisseur, ont
ete pris sur les MEDIANES MESUREES des vraies zones H1. Ils viennent donc des
donnees. Un seul couple de valeurs qui fonctionne ne vaut rien : c'est le
profil habituel d'un parametre ajuste. Un effet reel doit se retrouver sur
tout un voisinage, et varier regulierement.

CE QUE LA GRILLE MONTRE
Quinze couples, chacun confronte aux memes temoins par blocs, sur
l'echantillon complet puis sur chaque moitie. On regarde trois choses :
combien de couples sont positifs, s'ils forment une plage CONTINUE plutot
qu'un point isole, et si le maximum se trouve au milieu de cette plage ou
bien collee a un bord, ce qui signalerait que le vrai optimum est dehors.

    python test_placebo_grille.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from quant.mt5_loader import load_mt5_csv, resample
from quant.scalp_mtf import MTFConfig, backtest_mtf
from quant.surrogate import shuffle_blocks

N_TEMOINS = 4
GRAINE = 20261013
JAMBES = [1.2, 1.5, 1.8, 2.1, 2.5]
ZONES = [0.6, 0.9, 1.2]


def mesure(m1: pd.DataFrame, det: pd.DataFrame, jambe: float,
           zone: float) -> dict:
    cfg = MTFConfig(min_trend_force=0.0, partial_r=1e9, max_trades_per_day=50,
                    entry_mode="contact", tp_usd=10.0,
                    tf_minutes=60, atr_bars=24, max_hold_min=48 * 60,
                    ob_mode="placebo",
                    placebo_jambe_atr=jambe, placebo_zone_atr=zone)
    tr, st = backtest_mtf(det, m1, cfg)
    if tr.empty:
        return {}
    return {"n": len(tr), "jour": st["trades_par_jour"],
            "wr": st["winrate"] * 100, "R": tr.R.mean(),
            "sd": tr.R.std(ddof=1)}


def bloc(m1: pd.DataFrame, titre: str, graine: int) -> dict:
    jours = m1.index.normalize().nunique()
    print(f"\n{titre}  ({jours} jours)", flush=True)
    print("  recul  epaiss.     n   freq    WR     R reel   temoin"
          "   ecart     t")
    print("  " + "-" * 72, flush=True)
    det = resample(m1, "1h")
    tem = [shuffle_blocks(m1, 60, seed=graine + k) for k in range(N_TEMOINS)]
    tem_det = [resample(t, "1h") for t in tem]
    out = {}
    for jambe in JAMBES:
        for zone in ZONES:
            r = mesure(m1, det, jambe, zone)
            if not r or r["n"] < 30:
                print(f"  {jambe:<6.2f} {zone:<8.2f} {r.get('n', 0):>5}"
                      f"  (trop peu de trades)", flush=True)
                continue
            rs = [x for x in (mesure(t, td, jambe, zone)
                              for t, td in zip(tem, tem_det)) if x]
            rt = float(np.mean([x["R"] for x in rs])) if rs else np.nan
            ec = r["R"] - rt
            t = ec / (r["sd"] / np.sqrt(r["n"])) if r["sd"] > 0 else np.nan
            out[(jambe, zone)] = ec
            print(f"  {jambe:<6.2f} {zone:<8.2f} {r['n']:>5} {r['jour']:>5.2f} "
                  f"{r['wr']:5.1f}%  {r['R']:+.3f}  {rt:+.3f}  {ec:+.3f}  "
                  f"{t:+5.2f}", flush=True)
    return out


def carte(titre: str, d: dict) -> None:
    print(f"\n  {titre}")
    print("  recul \\ epaiss." + "".join(f"{z:>9.2f}" for z in ZONES))
    for jambe in JAMBES:
        ligne = f"  {jambe:>14.2f}"
        for zone in ZONES:
            v = d.get((jambe, zone), np.nan)
            ligne += "      --- " if np.isnan(v) else f"{v:>+9.3f}"
        print(ligne)


def main(chemin: str) -> None:
    t0 = time.time()
    m1 = resample(load_mt5_csv(chemin), "1min")
    print(f"{len(m1):,} bougies M1 | detection H1, zone posee par geometrie,")
    print("aucun Order Block. 'recul' = distance en ATR entre la fin de")
    print("l'impulsion et le bord de la zone. 'epaiss.' = epaisseur en ATR.")
    print("Le reglage mesure sur les vraies zones etait 1.80 et 0.90.")

    tout = bloc(m1, "ECHANTILLON COMPLET", GRAINE)
    coupe = m1.index[len(m1) // 2]
    a = bloc(m1[m1.index < coupe], "PREMIERE MOITIE", GRAINE + 100)
    b = bloc(m1[m1.index >= coupe], "SECONDE MOITIE", GRAINE + 200)

    print("\n\nCARTES DE L'ECART AU TEMOIN")
    carte("echantillon complet", tout)
    carte("premiere moitie", a)
    carte("seconde moitie", b)

    print("\n\nCOUPLES POSITIFS SUR LES DEUX MOITIES")
    bons = []
    for cle in tout:
        va, vb = a.get(cle, np.nan), b.get(cle, np.nan)
        if not np.isnan(va) and not np.isnan(vb) and va > 0 and vb > 0:
            bons.append((cle, tout[cle], va, vb))
    print(f"  {len(bons)} sur {len(tout)} couples testes.")
    for (j, z), nt, va, vb in sorted(bons, key=lambda x: -x[1]):
        print(f"    recul {j:.2f} ATR, epaisseur {z:.2f} ATR :"
              f" {nt:+.3f}   ({va:+.3f} puis {vb:+.3f})")
    if len(bons) <= 2:
        print("\n  Une ou deux cases isolees : c'est le profil d'un parametre")
        print("  ajuste, pas celui d'un effet reel.")
    else:
        print("\n  Verifier que ces couples se touchent sur la carte. Une plage")
        print("  continue est le signe d'un effet reel ; des cases dispersees")
        print("  restent du hasard, meme nombreuses.")
    print(f"\n  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
