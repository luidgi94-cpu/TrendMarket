"""Le moteur est-il honnete ? Verification sur un cas dont la reponse est connue.

LE SOUPCON
Dans toutes les mesures de la serie H1, le temoin lui-meme rapporte +0.48 a
+0.60 R par trade, avec une cible valant 1 a 2 fois le stop. Or un temoin
est, par construction, une suite de bougies dont l'ordre a ete detruit :
rien ne peut y etre anticipe. Toucher une cible a 2 R avant un stop a 1 R
doit y arriver une fois sur trois, ce qui donne une esperance NULLE. En
mesurer la moitie en plus signifie qu'une hypothese du moteur est trop
favorable, et que tous les R absolus de ce projet sont gonfles.

Un ecart au temoin peut rester valable malgre cela, puisque le biais frappe
les deux cotes. Mais un R absolu de +0.88 ne serait alors pas une prevision
de gain, et il ne faut pas le presenter comme telle.

LE TEST
Une marche aleatoire sans derive est fabriquee avec la volatilite de l'or,
puis donnee au moteur. La reponse est connue d'avance : l'esperance doit
etre NULLE, aux frais pres. Chaque minute est simulee en vingt sous-pas,
afin que les plus hauts et les plus bas des bougies soient realistes et non
pas reduits aux seules clotures.

  si le moteur rend environ zero   le moteur est sain, et le +0.5 R du
                                   temoin vient d'une propriete reelle des
                                   bougies d'or permutees, a identifier
  si le moteur rend nettement plus il y a un biais dans la mecanique de
                                   sortie, et les chiffres absolus de tout
                                   le projet sont a refaire

    python test_moteur_sain.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.scalp_mtf import MTFConfig, backtest_mtf

N_MIN = 588_000
SOUS_PAS = 20
SIGMA_MIN = 0.00035       # ecart-type par minute, calibre sur XAUUSD
DEPART = 3000.0


def marche_aleatoire(seed: int) -> pd.DataFrame:
    """M1 OHLC d'une marche aleatoire sans derive, hauts et bas realistes."""
    rng = np.random.default_rng(seed)
    pas = rng.normal(0.0, SIGMA_MIN / np.sqrt(SOUS_PAS),
                     size=(N_MIN, SOUS_PAS))
    chemin = np.cumsum(pas, axis=1)
    depart = np.concatenate([[0.0], np.cumsum(chemin[:, -1])[:-1]])
    chemin = chemin + depart[:, None]
    prix = DEPART * np.exp(chemin)
    ouv = DEPART * np.exp(depart)
    idx = pd.date_range("2025-01-02 00:00", periods=N_MIN, freq="1min",
                        tz="UTC")
    # L'OUVERTURE DOIT ENTRER DANS LE PLUS HAUT ET LE PLUS BAS. Sans cela,
    # une minute entierement descendante a un plus haut INFERIEUR a son
    # ouverture : la bougie est incoherente et son amplitude trop petite.
    # Le moteur teste ses barrieres sur le plus haut et le plus bas, donc des
    # amplitudes trop petites font manquer des sorties, et surtout le stop,
    # qui est le plus proche des deux. C'etait la cause du biais residuel de
    # +0.46 R que j'imputais a tort au moteur.
    return pd.DataFrame({
        "open": ouv,
        "high": np.maximum(prix.max(axis=1), ouv),
        "low": np.minimum(prix.min(axis=1), ouv),
        "close": prix[:, -1],
    }, index=idx)


def essai(m1: pd.DataFrame, nom: str, opts: dict) -> None:
    base = dict(min_trend_force=0.0, partial_r=1e9, max_trades_per_day=50,
                entry_mode="contact", tf_minutes=60, atr_bars=24,
                max_hold_min=48 * 60, ob_mode="placebo",
                placebo_jambe_atr=0.8, placebo_zone_atr=0.2)
    base.update(opts)
    tr, st = backtest_mtf(m1.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna(), m1, MTFConfig(**base))
    if tr.empty:
        print(f"  {nom:<26} aucun trade")
        return
    theo = 1.0 / (1.0 + tr.tp_en_r.median())
    sorties = tr["exit"].value_counts(normalize=True)
    print(f"  {nom:<26} {len(tr):>5} {st['winrate'] * 100:>6.1f}% "
          f"{tr.tp_en_r.median():>6.2f}R {100 * theo:>7.1f}% "
          f"{tr.R.mean():>+8.3f} {tr.R.std(ddof=1) / np.sqrt(len(tr)):>7.3f} "
          f"{100 * sorties.get('fin_fenetre', 0):>7.1f}%")


def main() -> None:
    print(f"Marche aleatoire SANS DERIVE, {N_MIN:,} minutes, "
          f"{SOUS_PAS} sous-pas par minute.")
    print("La reponse est connue : l'esperance doit etre nulle, aux frais")
    print("pres. 'WR theorique' est le taux de reussite qu'impose une marche")
    print("aleatoire pour la cible constatee, soit 1/(1+cible).")
    print("Si le winrate mesure depasse nettement le theorique, le moteur")
    print("surestime, et la derniere colonne dit si les sorties en fin de")
    print("fenetre en sont responsables.\n")
    print("  essai                          n      WR   cible  WR theo"
          "        R      se  fin fen.")
    print("  " + "-" * 86)

    for seed in (1, 2):
        m1 = marche_aleatoire(seed)
        print(f"  --- tirage {seed} "
              f"(volatilite horaire mediane "
              f"{(m1.high.resample('1h').max() - m1.low.resample('1h').min()).median():.2f} USD)")
        essai(m1, "cible 2 R, sans frais", dict(tp_usd=0.0, tp_r=2.0,
                                                spread_usd=0.0,
                                                slippage_usd=0.0))
        essai(m1, "cible 1 R, sans frais", dict(tp_usd=0.0, tp_r=1.0,
                                                spread_usd=0.0,
                                                slippage_usd=0.0))
        essai(m1, "cible 100 pips, sans frais", dict(tp_usd=10.0,
                                                     spread_usd=0.0,
                                                     slippage_usd=0.0))
        essai(m1, "cible 2 R, avec frais", dict(tp_usd=0.0, tp_r=2.0))

    print("\nLECTURE")
    print("  Un R significativement positif sur une marche aleatoire sans")
    print("  derive est impossible : ce serait de l'argent cree a partir de")
    print("  rien. Tout ecart de plus de deux erreurs-types au-dessus de zero")
    print("  designe donc un biais du moteur, pas une decouverte.")


if __name__ == "__main__":
    main()
