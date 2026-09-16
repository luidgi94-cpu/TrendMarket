"""Generateur de bougies M15 XAUUSD synthetiques.

Deux modes, et la distinction est le coeur de l'honnetete du backtest :

  mode="random"    : aucune structure intraday exploitable. Les breakouts ne
                     continuent pas plus souvent qu'ils n'echouent. Une
                     strategie ORB doit y perdre EXACTEMENT ses couts.
                     -> donne le winrate "plancher" et le taux de trades/jour,
                        deux chiffres qui ne dependent PAS d'un edge suppose.

  mode="structure" : on injecte une persistance de momentum pendant Londres
                     et NY (autocorrelation positive des rendements) et du
                     mean-reversion en Asie. C'est l'hypothese que l'ORB
                     cherche a exploiter. Le resultat mesure ce que la
                     strategie extrait SI l'effet existe - pas s'il existe.

Le profil de volatilite horaire, lui, est un fait empirique robuste sur l'or :
Asie calme, expansion a l'ouverture de Londres, pic sur le chevauchement
Londres/NY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Multiplicateur de volatilite par heure UTC (profil empirique de l'or)
HOUR_VOL = {
    0: 0.45, 1: 0.40, 2: 0.45, 3: 0.45, 4: 0.50, 5: 0.60,
    6: 0.85, 7: 1.25, 8: 1.55, 9: 1.50, 10: 1.35, 11: 1.30,
    12: 1.60, 13: 1.95, 14: 1.85, 15: 1.60, 16: 1.25,
    17: 0.95, 18: 0.85, 19: 0.80, 20: 0.70, 21: 0.55,
    22: 0.35, 23: 0.35,
}
LONDON_NY_HOURS = set(range(7, 17))
ASIA_HOURS = set(range(0, 6))


def generate_m15(n_days: int = 750, start_price: float = 4000.0,
                 daily_vol: float = 0.014, mode: str = "structure",
                 momentum_rho: float = 0.06, seed: int = 17) -> pd.DataFrame:
    """Retourne un DataFrame OHLC M15 indexe en UTC.

    daily_vol 1.4% sur or a 4000 USD -> ATR journalier ~55 USD, calibre sur
    les ordres de grandeur 2025-2026.
    momentum_rho : autocorrelation des rendements M15 en session active.
    0.06 est volontairement FAIBLE - un effet intraday realiste est tenu,
    pas spectaculaire.
    """
    rng = np.random.default_rng(seed)
    bars_per_day = 96
    idx = pd.date_range("2023-01-02", periods=n_days * bars_per_day,
                        freq="15min", tz="UTC")
    # on retire le week-end (marche ferme vendredi 21h -> dimanche 22h UTC)
    idx = idx[~((idx.dayofweek == 5) | ((idx.dayofweek == 4) & (idx.hour >= 21))
                | ((idx.dayofweek == 6) & (idx.hour < 22)))]

    hours = idx.hour.to_numpy()
    prof = np.array([HOUR_VOL[h] for h in hours])
    # normalise pour que la vol journaliere realisee colle a la cible
    bar_vol = daily_vol / np.sqrt(bars_per_day) * prof / np.sqrt((prof ** 2).mean())

    n = len(idx)
    shocks = rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)  # queues epaisses

    if mode == "structure":
        active = np.isin(hours, list(LONDON_NY_HOURS))
        asia = np.isin(hours, list(ASIA_HOURS))
        rho = np.where(active, momentum_rho, 0.0)
        rho = np.where(asia, -momentum_rho * 1.5, rho)  # Asie : mean-reversion
        eps = np.zeros(n)
        for t in range(1, n):
            eps[t] = rho[t] * eps[t - 1] + shocks[t]
        eps = eps / eps.std()
    elif mode == "random":
        eps = shocks / shocks.std()
    else:
        raise ValueError("mode : 'random' ou 'structure'")

    # drift journalier lent (tendance D1) pour que le biais quotidien existe
    day_id = (idx.normalize().astype("int64") // 10**9).to_numpy()
    uniq = np.unique(day_id)
    regime = np.zeros(len(uniq))
    t, sign = 0, 1.0
    while t < len(uniq):
        length = int(rng.gamma(4.0, 6.0)) + 3
        regime[t:t + length] = sign * abs(rng.normal(0.30, 0.15)) / 16.0
        sign = -sign if rng.random() < 0.6 else sign
        t += length
    drift_map = dict(zip(uniq, regime * daily_vol / bars_per_day * 16))
    drift = np.array([drift_map[d] for d in day_id])

    returns = drift + bar_vol * eps
    close = start_price * np.exp(np.cumsum(returns))

    # OHLC : range intra-bougie proportionnel a la vol locale
    span = np.abs(rng.normal(0, 1, n)) * bar_vol * close * 0.8
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) + span * rng.uniform(0.2, 1.0, n)
    low = np.minimum(open_, close) - span * rng.uniform(0.2, 1.0, n)

    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close}, index=idx)
