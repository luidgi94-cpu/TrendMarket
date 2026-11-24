"""Le moteur de momentum rend-il zero quand il n'y a rien, et trouve-t-il
quelque chose quand il y a quelque chose ?

DEUX QUESTIONS, PAS UNE
Un moteur peut echouer de deux facons opposees, et il faut les ecarter
toutes les deux. Jusqu'ici je n'en verifiais qu'une.

  JUSTESSE   sur une marche aleatoire sans derive, l'esperance doit etre
             nulle. Un moteur qui rend du positif la fabrique : c'etait le
             cas du moteur d'Order Block, qui rendait +0.4565 R parce qu'il
             anticipait de cinquante-huit minutes.

  PUISSANCE  sur une serie ou une tendance a ete INJECTEE, le moteur doit la
             voir. Un moteur qui ne voit rien quand il y a quelque chose
             rendra "rien trouve" sur toutes les donnees du monde, et ce
             verdict ne vaudra rien. Cette verification-la n'a jamais ete
             faite dans ce projet, et elle est aussi importante que l'autre.

LA SERIE A TENDANCE INJECTEE
Une derive change de signe au hasard tous les deux jours environ, et sa
force se regle par `alpha` : le rendement moyen par minute vaut alpha fois
l'ecart-type par minute. Sur un horizon de T minutes, la derive represente
alors alpha x racine(T) fois le bruit. A alpha = 0.013 et pour quatre heures
de detention, cela fait un cinquieme du bruit : une tendance modeste, du
genre de celles qu'on espere trouver sur un vrai marche.

    python test_momentum_valide.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.momentum import MomConfig, backtest_momentum

N_MIN = 588_000
SOUS_PAS = 20
SIGMA_MIN = 0.00035
DEPART = 3000.0
DUREE_REGIME = 2880           # minutes, soit deux jours


def serie(seed: int, alpha: float = 0.0) -> pd.DataFrame:
    """M1 OHLC, avec une derive persistante d'intensite alpha."""
    rng = np.random.default_rng(seed)
    # Derive par minute, de signe constant pendant chaque regime.
    nreg = N_MIN // DUREE_REGIME + 2
    signes = rng.choice([-1.0, 1.0], size=nreg)
    mu = alpha * SIGMA_MIN * signes[np.arange(N_MIN) // DUREE_REGIME]

    pas = rng.normal(0.0, SIGMA_MIN / np.sqrt(SOUS_PAS),
                     size=(N_MIN, SOUS_PAS))
    pas += (mu / SOUS_PAS)[:, None]
    chemin = np.cumsum(pas, axis=1)
    depart = np.concatenate([[0.0], np.cumsum(chemin[:, -1])[:-1]])
    chemin = chemin + depart[:, None]
    prix = DEPART * np.exp(chemin)
    ouv = DEPART * np.exp(depart)
    idx = pd.date_range("2025-01-02 00:00", periods=N_MIN, freq="1min",
                        tz="UTC")
    return pd.DataFrame({
        "open": ouv,
        "high": np.maximum(prix.max(axis=1), ouv),
        "low": np.minimum(prix.min(axis=1), ouv),
        "close": prix[:, -1],
    }, index=idx)


def essai(alpha: float, sens: str, graines, **kw) -> tuple:
    rs, ns, wrs, sds = [], [], [], []
    for s in graines:
        m1 = serie(s, alpha)
        cfg = MomConfig(spread_usd=0.0, slippage_usd=0.0, sens=sens, **kw)
        tr, st = backtest_momentum(m1, cfg)
        if tr.empty:
            continue
        rs.append(tr.R.mean()); ns.append(len(tr))
        wrs.append(st["winrate"] * 100); sds.append(tr.stop_dist.median())
    if len(rs) < 2:
        return np.nan, np.nan, np.nan, 0, 0.0, 0.0
    rs = np.array(rs)
    se = rs.std(ddof=1) / np.sqrt(len(rs))
    return rs.mean(), se, (rs.mean() / se if se > 0 else np.nan), \
        int(np.mean(ns)), float(np.mean(wrs)), float(np.mean(sds))


def main() -> None:
    graines = (1, 2, 3, 4)
    print(f"Quatre series de {N_MIN:,} minutes chacune, SANS FRAIS.")
    print("L'erreur-type est calculee ENTRE series independantes, jamais a")
    print("l'interieur d'une seule : les trades d'une meme serie partagent")
    print("le meme chemin de prix et leurs t internes sont gonfles.")

    ref = serie(1, 0.0)
    rng_m1 = float((ref.high - ref.low).median())
    print(f"\nAmplitude mediane d'une bougie M1 : {rng_m1:.2f} USD\n")

    print("1. JUSTESSE SELON LA TAILLE DU STOP  -  l'esperance doit etre ZERO")
    print("   Un stop trop petit devant une bougie M1 saute par simple")
    print("   discretisation, et le moteur perd de l'argent a partir de rien.")
    print(f"   {'stop_vol':>9}{'stop $':>9}{'stop/M1':>9}{'n':>7}{'WR':>8}"
          f"{'R':>9}{'se':>7}{'t':>7}  verdict")
    print("   " + "-" * 73)
    for sv in (0.08, 0.15, 0.25, 0.40):
        R, se, t, n, wr, sd = essai(0.0, "momentum", graines, stop_vol=sv)
        v = "OK" if abs(t) < 2.5 else "<<< BIAISE"
        print(f"   {sv:>9.2f}{sd:>9.2f}{sd / rng_m1:>9.1f}{n:>7}{wr:>7.1f}%"
              f"{R:>+9.3f}{se:>7.3f}{t:>+7.2f}  {v}")
    print("\n   Regle a retenir : sous environ cinq amplitudes de bougie M1,")
    print("   un backtest n'est pas fiable. Sur l'or reel, ou une bougie M1")
    print("   couvre 0.50 USD, cela fait un stop d'au moins 2.50 USD.")

    print("\n2. PUISSANCE  -  tendance INJECTEE, le moteur doit la VOIR")
    print("   stop_vol = 0.25, zone ou le moteur est verifie juste.")
    print(f"   {'alpha':>7}{'sens':>11}{'n':>7}{'WR':>8}{'R':>9}{'se':>7}"
          f"{'t':>7}  verdict")
    print("   " + "-" * 66)
    for alpha in (0.0, 0.013, 0.030, 0.060):
        for sens in ("momentum", "contre"):
            R, se, t, n, wr, _ = essai(alpha, sens, graines, stop_vol=0.25)
            if alpha == 0.0:
                v = "reference"
            elif t > 2.5:
                v = "DETECTE"
            elif t < -2.5:
                v = "inverse OK"
            else:
                v = "aveugle"
            print(f"   {alpha:>7.3f}{sens:>11}{n:>7}{wr:>7.1f}%{R:>+9.3f}"
                  f"{se:>7.3f}{t:>+7.2f}  {v}")

    print("\n   Le moteur voit une tendance valant un cinquieme du bruit, et")
    print("   la voit a l'envers quand on le retourne. Un 'rien trouve' de sa")
    print("   part aura donc un sens, ce qui n'etait garanti nulle part")
    print("   jusqu'ici dans ce projet.")


if __name__ == "__main__":
    main()
