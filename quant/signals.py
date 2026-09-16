"""Generation des forecasts de trend.

Principe : aucun signal unique n'est fiable. On combine deux familles
(momentum continu + breakout) sur trois horizons chacune, et on moyenne.
La moyenne d'horizons est ce qui evite l'over-fitting sur "le bon parametre",
qui n'existe pas hors-echantillon.

Toutes les series sont calculees en t et appliquees en t+1 : pas de lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SignalConfig


def daily_vol(prices: pd.Series, span: int, long_window: int,
              long_weight: float) -> pd.Series:
    """Vol quotidienne en % du prix, ancree sur sa moyenne longue.

    L'EWMA pur sous-estime la vol juste avant les chocs (elle vient de sortir
    d'une periode calme). On la melange avec une moyenne longue pour eviter
    de sur-lever a la veille d'un regime volatil.
    """
    returns = prices.pct_change()
    short = returns.ewm(span=span, min_periods=span).std()
    long = short.rolling(long_window, min_periods=span * 2).mean()
    blended = (1 - long_weight) * short + long_weight * long.fillna(short)
    return blended.replace(0.0, np.nan).ffill()


def _scale_forecast(raw: pd.Series, target: float, cap: float) -> pd.Series:
    """Normalise un forecast brut pour que |forecast| moyen == target.

    Le facteur d'echelle est calcule en expanding window (jamais sur
    l'ensemble de l'echantillon) pour rester hors-echantillon strict.
    """
    abs_mean = raw.abs().expanding(min_periods=252).mean()
    scalar = (target / abs_mean).replace([np.inf, -np.inf], np.nan)
    # borne le scalar : une periode de forecasts minuscules ne doit pas
    # produire un multiplicateur delirant
    scalar = scalar.clip(upper=scalar.rolling(512, min_periods=252).median() * 3)
    return (raw * scalar.ffill()).clip(-cap, cap)


def ewmac_forecast(prices: pd.Series, vol: pd.Series, fast: int, slow: int,
                   cfg: SignalConfig) -> pd.Series:
    """Croisement de moyennes exponentielles, normalise par la volatilite.

    Diviser par la vol est ce qui rend le signal comparable entre BTC (vol
    ~50%) et l'or (vol ~15%). Sans ca, BTC domine mecaniquement le book.
    """
    raw = prices.ewm(span=fast, min_periods=fast).mean() - \
          prices.ewm(span=slow, min_periods=slow).mean()
    normalised = raw / (prices * vol)
    return _scale_forecast(normalised, cfg.forecast_target, cfg.forecast_cap)


def breakout_forecast(prices: pd.Series, window: int,
                      cfg: SignalConfig) -> pd.Series:
    """Position dans le canal de Donchian, lissee.

    Sortie brute dans [-1, +1] : +1 au plus haut du canal, -1 au plus bas.
    Le lissage (window/4) evite de basculer sur une meche isolee.
    """
    roll_max = prices.rolling(window, min_periods=window // 2).max()
    roll_min = prices.rolling(window, min_periods=window // 2).min()
    mid = (roll_max + roll_min) / 2.0
    span = (roll_max - roll_min).replace(0.0, np.nan)
    raw = 2.0 * (prices - mid) / span
    smoothed = raw.ewm(span=max(window // 4, 2), min_periods=window // 4).mean()
    return _scale_forecast(smoothed, cfg.forecast_target, cfg.forecast_cap)


def combined_forecast(prices: pd.Series, vol: pd.Series,
                      cfg: SignalConfig) -> pd.Series:
    """Ensemble final, dans [-cap, +cap].

    Le forecast est CONTINU, pas binaire. Un trend faible = petite position.
    C'est la difference entre un systeme qui encaisse les faux departs et un
    qui se fait hacher dessus.
    """
    parts, weights = [], []

    for fast, slow in cfg.ewmac_pairs:
        parts.append(ewmac_forecast(prices, vol, fast, slow, cfg))
        weights.append(cfg.ewmac_weight / len(cfg.ewmac_pairs))

    for window in cfg.breakout_windows:
        parts.append(breakout_forecast(prices, window, cfg))
        weights.append((1.0 - cfg.ewmac_weight) / len(cfg.breakout_windows))

    stacked = pd.concat(parts, axis=1)
    blended = (stacked * np.asarray(weights)).sum(axis=1, min_count=1)

    # La moyenne de signaux correles reduit l'amplitude : on re-normalise
    # pour retrouver |forecast| moyen == target, puis on cappe.
    return _scale_forecast(blended, cfg.forecast_target, cfg.forecast_cap)


def build_forecasts(prices: pd.DataFrame, vols: pd.DataFrame,
                    cfg: SignalConfig) -> pd.DataFrame:
    """Forecast par instrument, aligne sur l'index prix."""
    return pd.DataFrame(
        {k: combined_forecast(prices[k], vols[k], cfg) for k in prices.columns},
        index=prices.index,
    )
