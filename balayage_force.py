"""Balayage du filtre de force de tendance.

La force de tendance est le seul filtre dont l'effet s'est revele monotone
sur les 527 jours de XAUUSD reel : le quartile faible perd de l'argent
(-0.085 R), le quartile fort en gagne (+0.108 R). Le seuil retenu jusqu'ici,
0.6, laisse donc passer des trades dont on sait qu'ils sont perdants.

Ce script mesure, pour chaque seuil, ce que l'on gagne et ce que l'on perd :
on gagne en esperance et en drawdown, on perd en frequence. Le but n'est pas
de trouver le seuil qui maximise une metrique, c'est de voir la forme de
l'arbitrage et de choisir en connaissance de cause.

CONTRE-MESURE OBLIGATOIRE
Chaque seuil est aussi mesure sur un TEMOIN construit en melangeant les
bougies M1 a l'interieur de chaque heure. Le temoin preserve le profil de
volatilite horaire et la forme des bougies, mais detruit l'enchainement.
Tout ce que le temoin reproduit n'est pas de l'edge, c'est de la geometrie.
Un seuil n'est retenu que si l'ECART au temoin se tient.

Sans cette contre-mesure, un balayage de parametre est une machine a
surajuster : sur seize seuils, le meilleur paraitra toujours bon.

    python balayage_force.py chemin/vers/XAUUSD_M1.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant.mt5_loader import load_mt5_csv, resample
from quant.scalp_mtf import MTFConfig, backtest_mtf
from quant.surrogate import shuffle_bars

SEUILS = [0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
N_TEMOINS = 5
GRAINE = 20260916


def drawdown_max(equity: pd.Series, capital: float) -> float:
    """Drawdown maximum en pourcentage du pic precedent."""
    if equity.empty:
        return 0.0
    courbe = pd.concat([pd.Series([capital]), equity], ignore_index=True)
    pic = courbe.cummax()
    return float(((courbe - pic) / pic).min() * 100)


def mesure(m5: pd.DataFrame, m1: pd.DataFrame, seuil: float,
           partiel: bool = True, stop_max: float | None = None) -> dict:
    """Mesure une configuration.

    partiel  : False desactive la prise partielle EN MEME TEMPS que la
               remontee du stop a l'equilibre, puisque le code les lie.
               On rend la cible partielle inatteignable plutot que de
               mettre sa fraction a zero : une fraction nulle laisserait
               quand meme le stop remonter a l'equilibre, ce qui n'est pas
               la variante que l'on veut mesurer.
    stop_max : filtre a posteriori sur la largeur du stop. ATTENTION, ce
               n'est pas equivalent a refuser le trade au moment ou il se
               presente : refuser un trade libere la place pour le
               suivant, ce que ce filtre ne reproduit pas. C'est une
               premiere estimation de l'effet sur le R par trade, pas une
               simulation exacte.
    """
    cfg = MTFConfig(min_trend_force=seuil,
                    partial_r=1e9 if not partiel else 1.0)
    tr, st = backtest_mtf(m5, m1, cfg)
    if stop_max is not None and not tr.empty:
        tr = tr[tr.stop_dist <= stop_max].copy()
        if not tr.empty:
            tr["equity"] = cfg.capital + tr.pnl.cumsum()
            jours = m5.index.normalize().nunique()
            st = {"trades_par_jour": len(tr) / jours if jours else 0.0,
                  "winrate": (tr.pnl > 0).mean(),
                  "capital_final": float(tr.equity.iloc[-1])}
    if tr.empty:
        return {"trades": 0, "par_jour": 0.0, "winrate": np.nan,
                "R_moyen": np.nan, "R_ecart": np.nan, "dd_max": 0.0,
                "final": cfg.capital}
    return {"trades": len(tr),
            "par_jour": st["trades_par_jour"],
            "winrate": st["winrate"] * 100,
            "R_moyen": tr.R.mean(),
            "R_ecart": tr.R.std(ddof=1),
            "dd_max": drawdown_max(tr.equity, cfg.capital),
            "final": st["capital_final"]}


def main(chemin: str) -> None:
    brut = load_mt5_csv(chemin)
    m1 = resample(brut, "1min")
    m5 = resample(brut, "5min")
    jours = m5.index.normalize().nunique()
    print(f"Donnees : {len(m1):,} bougies M1  |  {jours} jours  "
          f"|  {m5.index[0]:%Y-%m-%d} a {m5.index[-1]:%Y-%m-%d}\n")

    # --- temoins : on les construit une fois et on les reutilise pour tous
    # --- les seuils, sinon on compare des seuils sur des bruits differents.
    print(f"Construction de {N_TEMOINS} temoins par permutation horaire...")
    temoins = []
    for k in range(N_TEMOINS):
        t1 = shuffle_bars(m1, seed=GRAINE + k)
        temoins.append((resample(t1, "5min"), t1))
    print("fait.\n")

    lignes = []
    for s in SEUILS:
        reel = mesure(m5, m1, s)
        rs = [mesure(t5, t1, s) for t5, t1 in temoins]
        rs = [r for r in rs if r["trades"] > 0]
        r_temoin = float(np.mean([r["R_moyen"] for r in rs])) if rs else np.nan
        n_temoin = float(np.mean([r["par_jour"] for r in rs])) if rs else np.nan
        ecart = reel["R_moyen"] - r_temoin if rs else np.nan

        # Test de Welch du reel contre la moyenne des temoins : l'incertitude
        # qui compte est celle de l'echantillon reel, la seule qu'on tradera.
        t_stat = np.nan
        if reel["trades"] > 1 and not np.isnan(ecart):
            err = reel["R_ecart"] / np.sqrt(reel["trades"])
            t_stat = ecart / err if err > 0 else np.nan

        lignes.append({"seuil": s, **reel, "R_temoin": r_temoin,
                       "jour_temoin": n_temoin, "ecart": ecart, "t": t_stat})
        print(f"  seuil {s:<4} : {reel['trades']:>5} trades  "
              f"{reel['par_jour']:>5.2f}/j  "
              f"WR {reel['winrate']:>5.1f}%  "
              f"R {reel['R_moyen']:+.3f}  "
              f"temoin {r_temoin:+.3f}  "
              f"ecart {ecart:+.3f}  "
              f"t {t_stat:>5.2f}  "
              f"DD {reel['dd_max']:>6.1f}%")

    df = pd.DataFrame(lignes)
    sortie = Path("resultats_balayage_force.csv")
    df.to_csv(sortie, index=False)
    print(f"\nTableau complet -> {sortie}")

    # --- variantes imposees par la taille de compte -----------------------
    # Le lot minimum du courtier est 0.01. Il n'existe pas de fraction de
    # 0.01 lot, donc la prise partielle du systeme valide est IMPOSSIBLE a
    # executer. La variante sans partiel n'est pas un raffinement : c'est
    # la seule reellement tradable. Elle doit donc etre mesuree, contre son
    # temoin comme le reste.
    base = df.loc[df.ecart.idxmax(), "seuil"] if not df.ecart.isna().all() else 0.6
    print(f"\nVARIANTES au seuil {base}")
    print("  (le partiel est impossible a 0.01 lot ; le filtre de stop borne")
    print("   le risque en pourcentage du capital)\n")

    for nom, part, smax in [("partiel + sans filtre", True,  None),
                            ("partiel + stop <= 7",   True,  7.0),
                            ("SANS partiel",          False, None),
                            ("SANS partiel + stop<=7", False, 7.0)]:
        r = mesure(m5, m1, base, partiel=part, stop_max=smax)
        rs = [mesure(t5, t1, base, partiel=part, stop_max=smax)
              for t5, t1 in temoins]
        rs = [x for x in rs if x["trades"] > 0]
        rt = float(np.mean([x["R_moyen"] for x in rs])) if rs else np.nan
        ec = r["R_moyen"] - rt if rs else np.nan
        t_stat = np.nan
        if r["trades"] > 1 and not np.isnan(ec):
            err = r["R_ecart"] / np.sqrt(r["trades"])
            t_stat = ec / err if err > 0 else np.nan
        print(f"  {nom:<24} {r['trades']:>5} trades  {r['par_jour']:>5.2f}/j  "
              f"WR {r['winrate']:>5.1f}%  R {r['R_moyen']:+.3f}  "
              f"temoin {rt:+.3f}  ecart {ec:+.3f}  t {t_stat:>5.2f}  "
              f"DD {r['dd_max']:>6.1f}%")

    print("\n  Le filtre de stop est applique A POSTERIORI : il retire les")
    print("  trades a stop large sans liberer la place pour ceux qui")
    print("  auraient suivi. L'effet reel sur la frequence est donc")
    print("  sous-estime. Traiter ce chiffre comme une borne, pas comme")
    print("  une mesure.")

    print("\nLECTURE")
    print("  - 'ecart' est la seule colonne qui mesure de l'edge. 'R_moyen'")
    print("    seul recompense la geometrie autant que le signal.")
    print("  - Un t inferieur a 2 ne permet pas de conclure, quel que soit")
    print("    le R affiche.")
    print("  - Le meilleur seuil du tableau est en partie le fruit du hasard :")
    print("    sur neuf essais, le maximum est biaise vers le haut. Prendre")
    print("    un seuil ou le voisinage tient aussi, pas un pic isole.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
