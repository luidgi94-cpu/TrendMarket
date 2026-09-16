"""Overlays conditionnels : funding perp (BTC/ETH) et macro or (XAU/XAG).

Un overlay ne cree jamais un trade. Il module la taille d'un trade que le
trend a deja valide. C'est une distinction importante : chaque filtre
autorise a ouvrir des positions multiplie les chemins d'over-fitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OverlayConfig


def funding_multiplier(funding_annual: pd.Series, forecast: pd.Series,
                       cfg: OverlayConfig) -> pd.Series:
    """Module le LONG crypto selon le funding rate du perpetuel.

    Lecture institutionnelle : le funding est la mesure la plus propre du
    positionnement a effet de levier accessible en retail. Funding tres
    positif = les longs paient pour tenir = le trade est surcharge =
    risque de cascade de liquidations sur le premier choc.

    On ALLEGE le long, on ne l'inverse pas : le funding est un indicateur
    de fragilite, pas de direction. Shorter un trend haussier parce que le
    funding est chaud est la facon classique de se faire sortir.

    Symetriquement, funding negatif en trend haussier = on est PAYE pour
    tenir la position : taille pleine.
    """
    mult = pd.Series(1.0, index=forecast.index, dtype=float)
    if funding_annual is None or funding_annual.dropna().empty:
        return mult

    f = funding_annual.reindex(forecast.index).ffill()
    # lissage 3j : le funding instantane est bruite, la persistance compte
    f_smooth = f.rolling(3, min_periods=1).mean()

    crowded_long = (f_smooth > cfg.funding_hot_annual) & (forecast > 0)
    crowded_short = (f_smooth < -cfg.funding_hot_annual) & (forecast < 0)
    mult[crowded_long | crowded_short] = 1.0 - cfg.funding_haircut
    return mult


def gold_macro_multiplier(dxy: pd.Series, real_yield_proxy: pd.Series,
                          forecast: pd.Series, cfg: OverlayConfig) -> pd.Series:
    """Module l'or selon le dollar et les taux longs US.

    L'or ne verse aucun coupon : son prix est un cout d'opportunite contre
    les taux reels. Un trend or haussier contre des taux reels qui montent
    et un dollar qui se renforce est un trend de mauvaise qualite.

    On n'interdit pas le trade (le trend a parfois raison contre la macro),
    on reduit la taille. Confirmation macro -> bonus modere.
    """
    mult = pd.Series(1.0, index=forecast.index, dtype=float)
    if dxy is None or dxy.dropna().empty:
        return mult

    d = dxy.reindex(forecast.index).ffill()
    dxy_trend = np.sign(d - d.rolling(cfg.dxy_window, min_periods=30).mean())

    if real_yield_proxy is not None and not real_yield_proxy.dropna().empty:
        y = real_yield_proxy.reindex(forecast.index).ffill()
        yield_trend = np.sign(y - y.rolling(cfg.dxy_window, min_periods=30).mean())
    else:
        yield_trend = pd.Series(0.0, index=forecast.index)

    direction = np.sign(forecast)
    # vent de face : long or + dollar fort + taux longs en hausse
    headwind = (direction > 0) & (dxy_trend > 0) & (yield_trend > 0)
    tailwind = (direction > 0) & (dxy_trend < 0) & (yield_trend < 0)
    mult[headwind.fillna(False)] = 0.6
    mult[tailwind.fillna(False)] = 1.2
    return mult


def apply_overlays(forecasts: pd.DataFrame, funding: pd.DataFrame | None,
                   macro: pd.DataFrame | None, cfg: OverlayConfig
                   ) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Applique les overlays disponibles. Absence de donnee = multiplicateur 1."""
    out = forecasts.copy()
    applied: dict[str, pd.Series] = {}

    if cfg.use_funding and funding is not None:
        for key in ("BTC", "ETH"):
            if key in out.columns and key in funding.columns:
                m = funding_multiplier(funding[key], out[key], cfg)
                out[key] = out[key] * m
                applied[f"funding_{key}"] = m

    if cfg.use_gold_macro and macro is not None:
        dxy = macro.get("DXY")
        yld = macro.get("UST10Y")
        for key in ("XAU", "XAG"):
            if key in out.columns:
                m = gold_macro_multiplier(dxy, yld, out[key], cfg)
                out[key] = out[key] * m
                applied[f"macro_{key}"] = m

    return out, applied
