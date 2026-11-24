"""Momentum, repli, et entree a l'ordre LIMITE.

D'OU VIENT CETTE IDEE
Deux mesures la designent. D'abord, sur les bandes de VWAP, le seul sens
qui batte son temoin est la cassure, +0.104 R : l'or prolonge ses mouvements
intraday au lieu de revenir a sa moyenne. Ensuite, cette meme cassure perd
quand meme de l'argent, parce que son temoin vaut -0.198 R : le pari coute
0.198 R avant toute lecture du marche. La raison est mecanique, un ordre
stop vous remplit APRES le depart du mouvement, donc au plus mauvais prix de
la bougie, alors qu'un ordre limite vous remplit a votre prix ou mieux.

D'ou la forme a essayer : garder la direction du momentum, et la jouer avec
un ordre limite pose sur un repli, et non avec un ordre stop sur une
cassure. Acheter un creux dans une tendance haussiere.

CE QUI EST FAIT POUR NE PAS REPRODUIRE L'ERREUR PRECEDENTE
Le moteur d'Order Block anticipait de cinquante-huit minutes parce qu'il
lisait une bougie H1 etiquetee par son bord GAUCHE tout en utilisant son
prix de cloture. Ici il n'y a AUCUN resample et aucune unite de detection :
tendance, volatilite et extreme recent sont des moyennes glissantes sur les
bougies M1, toutes decalees d'une bougie. Le niveau d'ordre de la bougie j
ne depend donc que des bougies jusqu'a j-1, et seule la bougie j decide du
remplissage.

LE PARI
  tendance    signe de l'ecart entre deux moyennes exponentielles longues
  repli       niveau pose a `repli_atr` sous l'extreme recent, pour un achat
  entree      ordre limite a ce niveau, valable `validite_min` minutes,
              arme seulement si le prix est encore AU-DESSUS du niveau
  stop        `stop_atr` au-dela du niveau
  cible       proportionnelle au risque, ou fixe en dollars l'once
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MomConfig:
    spread_usd: float = 0.15
    slippage_usd: float = 0.10
    units: float = 1.0

    ema_rapide: int = 1200        # minutes, soit 20 heures
    ema_lente: int = 3600         # minutes, soit 60 heures
    # UNITE DE VOLATILITE : l'etendue du prix sur `vol_bars` minutes, soit
    # le plus haut moins le plus bas de la journee ecoulee. Surtout PAS une
    # moyenne des amplitudes de bougies M1 : celle-ci vaut une MINUTE de
    # volatilite et non une journee. Avec elle, un stop regle a 0.4 valait
    # 0.81 USD quand une bougie M1 en couvre 1.85, si bien que le stop
    # sautait par simple discretisation. Le moteur rendait alors -0.193 R
    # sur une marche aleatoire, ou la reponse doit etre zero.
    vol_bars: int = 1440          # minutes, soit 24 heures
    extreme_bars: int = 240       # fenetre de l'extreme recent, en minutes

    # Tous deux en FRACTION de cette etendue.
    repli_vol: float = 0.12       # profondeur du repli
    stop_vol: float = 0.08        # stop au-dela du niveau
    validite_min: int = 240       # duree de vie de l'ordre, en minutes

    cible_r: float = 2.0          # cible en multiples du risque
    tp_usd: float = 0.0           # si > 0, cible fixe en dollars l'once
    duree_max_min: int = 1440
    max_trades_jour: int = 10
    heures: tuple[int, int] = (0, 24)
    sens: str = "momentum"        # "momentum" | "contre"


def _etendue(df: pd.DataFrame, w: int) -> np.ndarray:
    """Etendue du prix sur w minutes : plus haut moins plus bas."""
    hi = df.high.rolling(w, min_periods=w // 2).max()
    lo = df.low.rolling(w, min_periods=w // 2).min()
    return (hi - lo).to_numpy()


def backtest_momentum(m1: pd.DataFrame, cfg: MomConfig):
    o = m1.open.to_numpy(); h = m1.high.to_numpy()
    l = m1.low.to_numpy(); c = m1.close.to_numpy()
    n = len(m1)

    close = m1.close
    ecart = (close.ewm(span=cfg.ema_rapide).mean()
             - close.ewm(span=cfg.ema_lente).mean())
    vol = _etendue(m1, cfg.vol_bars)
    haut = m1.high.rolling(cfg.extreme_bars, min_periods=cfg.extreme_bars // 2).max()
    bas = m1.low.rolling(cfg.extreme_bars, min_periods=cfg.extreme_bars // 2).min()

    # DECALAGE D'UNE BOUGIE sur tout ce qui sert a DECIDER. La bougie j ne
    # sert qu'a tester le remplissage, jamais a calculer le niveau.
    d_arr = np.sign(ecart.shift(1).to_numpy())
    vol_d = np.roll(vol, 1); vol_d[0] = np.nan
    haut_d = haut.shift(1).to_numpy()
    bas_d = bas.shift(1).to_numpy()
    close_d = np.roll(c, 1); close_d[0] = np.nan

    heures = m1.index.hour.to_numpy()
    jours = m1.index.normalize()
    ouvert = (heures >= cfg.heures[0]) & (heures < cfg.heures[1])

    trades = []
    par_jour = {}
    j = max(cfg.ema_lente, cfg.vol_bars, cfg.extreme_bars) + 2
    while j < n - 1:
        if not ouvert[j] or np.isnan(vol_d[j]) or vol_d[j] <= 0 \
                or np.isnan(d_arr[j]) or d_arr[j] == 0 \
                or np.isnan(haut_d[j]) or np.isnan(bas_d[j]):
            j += 1; continue
        jr = jours[j]
        if par_jour.get(jr, 0) >= cfg.max_trades_jour:
            j += 1; continue

        d = int(d_arr[j])
        if cfg.sens == "contre":
            d = -d
        a = vol_d[j]
        # Niveau de repli : sous l'extreme recent pour un achat, au-dessus
        # pour une vente.
        niveau = (haut_d[j] - cfg.repli_vol * a) if d > 0 \
            else (bas_d[j] + cfg.repli_vol * a)
        # L'ordre n'est arme que si le prix doit encore VENIR au niveau.
        if (close_d[j] <= niveau) if d > 0 else (close_d[j] >= niveau):
            j += 1; continue

        # Attente du remplissage, dans la limite de validite de l'ordre.
        fin_ordre = min(j + cfg.validite_min, n - 1)
        touche = -1
        for t in range(j, fin_ordre):
            if (l[t] <= niveau) if d > 0 else (h[t] >= niveau):
                touche = t; break
        if touche < 0:
            j = fin_ordre; continue

        # Ordre limite : rempli au niveau, ou a l'ouverture si elle est deja
        # au-dela, ce qui donne un meilleur prix. C'est la mecanique reelle.
        entree = min(niveau, o[touche]) if d > 0 else max(niveau, o[touche])
        risque = cfg.stop_vol * a + abs(entree - niveau)
        if risque <= 0:
            j = touche + 1; continue
        stop = entree - d * risque
        cible = entree + d * (cfg.tp_usd if cfg.tp_usd > 0
                              else cfg.cible_r * risque)
        cible_r = (cible - entree) * d / risque

        fin = touche
        sortie, pourquoi = 0.0, "fin_fenetre"
        if (l[touche] <= stop) if d > 0 else (h[touche] >= stop):
            sortie, pourquoi = -1.0, "stop"
        else:
            vu = False
            for t in range(touche + 1, min(touche + cfg.duree_max_min, n)):
                fin = t
                if (l[t] <= stop) if d > 0 else (h[t] >= stop):
                    sortie, pourquoi, vu = -1.0, "stop", True
                    break
                if (h[t] >= cible) if d > 0 else (l[t] <= cible):
                    sortie, pourquoi, vu = cible_r, "cible", True
                    break
            if not vu:
                sortie = (c[fin] - entree) * d / risque

        cout = (cfg.spread_usd + cfg.slippage_usd) * cfg.units
        pnl = sortie * risque * cfg.units - cout
        par_jour[jr] = par_jour.get(jr, 0) + 1
        trades.append({"ts": m1.index[touche], "date": jr,
                       "sens": "long" if d > 0 else "short",
                       "entree": entree, "stop_dist": risque,
                       "cible_r": cible_r, "R": pnl / (risque * cfg.units),
                       "pnl": pnl, "sortie": pourquoi, "i": touche,
                       "i_fin": fin})
        j = fin + 1

    df = pd.DataFrame(trades)
    nj = pd.Index(jours).nunique()
    st = {"jours": nj, "nb_trades": len(df),
          "trades_par_jour": len(df) / nj if nj else 0.0}
    if not df.empty:
        st |= {"winrate": (df.pnl > 0).mean(), "R_moyen": df.R.mean(),
               "stop_median": df.stop_dist.median(),
               "cible_r_median": df.cible_r.median()}
    return df, st
