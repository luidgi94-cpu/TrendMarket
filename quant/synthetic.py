"""Generateur de donnees synthetiques, pour VALIDER LE MOTEUR.

A lire attentivement : un backtest sur donnees synthetiques ne prouve
AUCUN edge. Il sert a deux choses precises :

1. regime "trend"  -> le systeme doit gagner. Valide le cablage du signal.
2. regime "random" -> marche aleatoire sans trend. Le systeme doit perdre
   environ le montant des couts, et RIEN DE PLUS. Si un backtest est
   profitable sur une marche aleatoire, il y a du lookahead dans le code.
   C'est le test le plus important du fichier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# vol annualisee et classe, calibrees sur les ordres de grandeur reels
PROFILES = {
    "BTC": (0.50, "crypto"), "ETH": (0.65, "crypto"),
    "XAU": (0.16, "metal"),  "XAG": (0.28, "metal"),
    "SPX": (0.17, "equity"), "WTI": (0.35, "energy"),
    "EUR": (0.08, "fx"),     "UST": (0.06, "rates"),
}
CLASS_LOADING = {"crypto": 0.35, "metal": 0.25, "equity": 0.55,
                 "energy": 0.40, "fx": 0.30, "rates": -0.25}


def generate(keys: list[str], n_days: int = 4000, mode: str = "trend",
             seed: int = 7, regime_days: float = 200.0,
             trend_sharpe: float = 0.40) -> pd.DataFrame:
    """regime_days : duree moyenne d'un regime de trend, en jours ouvres.
    Calibre a ~200j (9-10 mois) d'apres la litterature time-series momentum
    (Moskowitz/Ooi/Pedersen) : le momentum 12 mois predit le mois suivant.
    trend_sharpe : Sharpe ANNUALISE du drift a l'interieur d'un regime.
    C'est le plafond theorique qu'un oracle parfait atteindrait.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    classes = sorted({PROFILES[k][1] for k in keys})

    # --- structure de correlation : facteur global + facteurs de classe ---
    market = rng.standard_normal(n_days)
    class_f = {c: rng.standard_normal(n_days) for c in classes}

    # --- clustering de volatilite (GARCH-like) ---------------------------
    def vol_path() -> np.ndarray:
        v = np.empty(n_days)
        v[0] = 1.0
        shock = rng.standard_normal(n_days)
        for t in range(1, n_days):
            v[t] = np.sqrt(0.02 + 0.10 * (v[t - 1] * shock[t - 1]) ** 2
                           + 0.87 * v[t - 1] ** 2)
        return v / v.mean()

    out = {}
    for key in keys:
        ann_vol, cls = PROFILES[key]
        daily = ann_vol / np.sqrt(256)

        beta_m = CLASS_LOADING[cls] * 0.6
        beta_c = 0.55
        idio = rng.standard_normal(n_days)
        common = beta_m * market + beta_c * class_f[cls]
        mix = common + np.sqrt(max(1 - beta_m**2 - beta_c**2, 0.10)) * idio
        mix = mix / mix.std()

        # queues epaisses : les rendements reels ne sont pas gaussiens
        jumps = rng.standard_t(df=4, size=n_days) / np.sqrt(2.0)
        shocks = 0.75 * mix + 0.25 * jumps

        if mode == "trend":
            # drift a changements de regime : duree moyenne ~4 mois, ce qui
            # correspond a la persistance empirique des trends macro
            drift = np.zeros(n_days)
            t, sign = 0, rng.choice([-1.0, 1.0])
            while t < n_days:
                length = int(rng.gamma(shape=4.0, scale=regime_days / 4.0)) + 20
                # Sharpe ANNUALISE du drift a l'interieur d'un regime.
                # Calibre bas (~0.4) : c'est l'ordre de grandeur empirique
                # des trends macro. Un drift plus fort produirait des
                # Sharpe de backtest fantaisistes et ne validerait rien.
                strength = abs(rng.normal(trend_sharpe, trend_sharpe / 2)) / 16.0
                drift[t:t + length] = sign * strength * daily
                sign = -sign if rng.random() < 0.65 else sign
                t += length
        elif mode == "random":
            drift = np.zeros(n_days)  # aucune structure exploitable
        else:
            raise ValueError("mode doit etre 'trend' ou 'random'")

        returns = drift + daily * vol_path() * shocks
        out[key] = 100.0 * np.exp(np.cumsum(returns))

    return pd.DataFrame(out, index=dates)


def synthetic_funding(prices: pd.DataFrame, keys: list[str],
                      seed: int = 11) -> pd.DataFrame:
    """Funding synthetique correle au momentum recent.

    Reproduit le fait stylise reel : apres une hausse, les longs s'entassent
    et le funding devient positif.
    """
    rng = np.random.default_rng(seed)
    cols = {}
    for key in keys:
        if key not in prices.columns:
            continue
        mom = prices[key].pct_change(20).fillna(0.0)
        base = (mom / mom.std()).clip(-3, 3) * 0.12
        noise = rng.normal(0, 0.05, len(prices))
        cols[key] = pd.Series(base.to_numpy() + noise, index=prices.index) + 0.08
    return pd.DataFrame(cols)
