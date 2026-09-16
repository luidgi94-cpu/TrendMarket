"""Moteur de backtest vectorise, net de couts.

Regle non negociable : les positions calculees avec l'information de la
cloture t sont appliquees aux rendements de t+1. Tout backtest qui ne fait
pas ca produit des Sharpe fantaisistes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BY_KEY, Config
from .overlays import apply_overlays
from .portfolio import apply_buffering, target_notionals
from .signals import build_forecasts, daily_vol


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series           # rendements quotidiens nets
    gross_returns: pd.Series     # avant couts
    positions: pd.DataFrame      # notionnels tenus
    forecasts: pd.DataFrame
    pnl_by_asset: pd.DataFrame   # PnL net en devise, par instrument
    costs: pd.Series
    idm: pd.Series
    turnover: pd.Series
    config: Config


def run_backtest(prices: pd.DataFrame, cfg: Config,
                 funding: pd.DataFrame | None = None,
                 macro: pd.DataFrame | None = None) -> BacktestResult:
    keys = [k for k in cfg.keys if k in prices.columns]
    prices = prices[keys].astype(float).sort_index()
    returns = prices.pct_change()

    vols = pd.DataFrame(
        {k: daily_vol(prices[k], cfg.risk.vol_ewma_span,
                      cfg.risk.vol_long_window, cfg.risk.vol_long_weight)
         for k in keys},
        index=prices.index,
    )

    forecasts = build_forecasts(prices, vols, cfg.signal)
    forecasts, _ = apply_overlays(forecasts, funding, macro, cfg.overlay)

    target, idm, unit = target_notionals(
        forecasts, vols, returns, cfg.capital,
        cfg.signal.forecast_target, cfg.risk,
    )
    held = apply_buffering(target, unit, cfg.risk)

    # --- decalage anti-lookahead -------------------------------------------
    # position decidee a la cloture t -> subit le rendement de t+1
    active = held.shift(1).fillna(0.0)

    pnl_gross = active * returns.fillna(0.0)

    # --- couts --------------------------------------------------------------
    # cost_bps est un aller-retour ; un trade simple coute la moitie.
    traded = held.diff().abs().fillna(held.abs())
    one_way_bps = pd.Series({k: BY_KEY[k].cost_bps / 2.0 for k in keys})
    cost_by_asset = traded.mul(one_way_bps / 10_000.0, axis=1)

    pnl_net = pnl_gross - cost_by_asset

    daily_net = pnl_net.sum(axis=1) / cfg.capital
    daily_gross = pnl_gross.sum(axis=1) / cfg.capital
    costs = cost_by_asset.sum(axis=1) / cfg.capital

    equity = cfg.capital * (1.0 + daily_net).cumprod()
    turnover = traded.sum(axis=1) / cfg.capital

    return BacktestResult(
        equity=equity,
        returns=daily_net,
        gross_returns=daily_gross,
        positions=active,
        forecasts=forecasts,
        pnl_by_asset=pnl_net,
        costs=costs,
        idm=idm,
        turnover=turnover,
        config=cfg,
    )


def extract_trades(positions: pd.DataFrame,
                   pnl_by_asset: pd.DataFrame) -> pd.DataFrame:
    """Decoupe le PnL en 'trades' : periodes de signe constant et non nul.

    Le winrate par trade n'est pas un objectif, c'est un diagnostic. Un trend
    follower sain affiche 35-45% : beaucoup de faux departs coupes petit,
    quelques trends tenus longtemps. Un winrate eleve ici serait le signe
    qu'on coupe les gagnants trop tot.
    """
    rows = []
    for key in positions.columns:
        pos = positions[key].to_numpy()
        pnl = pnl_by_asset[key].to_numpy()
        idx = positions.index
        sign = np.sign(pos)
        start = None
        for t in range(len(sign)):
            if sign[t] != 0 and start is None:
                start = t
            elif start is not None and (sign[t] != sign[start] or t == len(sign) - 1):
                end = t
                rows.append({
                    "asset": key,
                    "direction": "long" if sign[start] > 0 else "short",
                    "entry": idx[start],
                    "exit": idx[end],
                    "bars": end - start,
                    "pnl": float(np.nansum(pnl[start:end + 1])),
                })
                start = t if sign[t] != 0 else None
    return pd.DataFrame(rows)
