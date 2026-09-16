#!/usr/bin/env python3
"""Tests de validation du moteur. A lancer avant toute lecture de resultat.

Le test decisif est test_no_edge_on_random_walk : sur une marche aleatoire
sans structure, le systeme DOIT perdre a peu pres ses couts. S'il gagne,
c'est qu'il y a du lookahead quelque part et tout le reste est invalide.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant import metrics
from quant.backtest import run_backtest
from quant.config import ALL_KEYS, Config
from quant.synthetic import generate

FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if condition else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
    if not condition:
        FAILED.append(name)


def test_no_lookahead_shift():
    """La position appliquee en t doit etre celle DECIDEE en t-1."""
    prices = generate(["BTC", "XAU"], n_days=900, mode="trend", seed=3)
    res = run_backtest(prices, Config(keys=["BTC", "XAU"]))
    # correlation entre position active et rendement du MEME jour :
    # si le moteur trichait, elle serait fortement positive
    joint = pd.concat([res.positions["BTC"], prices["BTC"].pct_change()],
                      axis=1).dropna()
    same_day = joint.iloc[:, 0].corr(joint.iloc[:, 1])
    check("pas de correlation position/rendement du jour",
          abs(same_day) < 0.15, f"corr = {same_day:.3f}")


def test_no_edge_on_random_walk():
    """TEST DECISIF. Aucune structure -> aucun profit possible."""
    sharpes = []
    for seed in range(8):
        prices = generate(ALL_KEYS, n_days=2500, mode="random", seed=seed)
        res = run_backtest(prices, Config(keys=ALL_KEYS))
        s = metrics.summary(res.returns, res.equity)
        sharpes.append(s["Sharpe"])
    mean_sharpe = float(np.mean(sharpes))
    check("Sharpe ~0 sur marche aleatoire (8 seeds)",
          -0.6 < mean_sharpe < 0.35,
          f"Sharpe moyen = {mean_sharpe:.2f}, par seed = "
          f"{[round(x, 2) for x in sharpes]}")


def test_vol_targeting_works():
    """La vol realisee doit atterrir pres de la cible, sinon le sizing ment."""
    prices = generate(ALL_KEYS, n_days=3000, mode="trend", seed=5)
    cfg = Config(keys=ALL_KEYS)
    res = run_backtest(prices, cfg)
    realised = res.returns.std() * np.sqrt(256)
    check("vol realisee proche de la cible",
          0.5 * cfg.risk.vol_target < realised < 2.0 * cfg.risk.vol_target,
          f"cible {cfg.risk.vol_target:.0%}, realisee {realised:.1%}")


def test_risk_balance_btc_vs_gold():
    """BTC (vol ~50%) et l'or (vol ~16%) doivent porter un risque comparable.

    C'est tout l'interet du vol targeting : sans lui, BTC represente
    ~75% du risque du book et l'or n'est que de la decoration.
    """
    prices = generate(["BTC", "XAU"], n_days=3000, mode="trend", seed=9)
    res = run_backtest(prices, Config(keys=["BTC", "XAU"]))
    vol_btc = res.pnl_by_asset["BTC"].std()
    vol_xau = res.pnl_by_asset["XAU"].std()
    ratio = vol_btc / vol_xau
    check("risque BTC/XAU equilibre", 0.5 < ratio < 2.0,
          f"ratio de vol du PnL = {ratio:.2f}")


def test_buffering_cuts_turnover():
    """Le buffering doit reduire nettement le turnover."""
    from quant.config import RiskConfig
    prices = generate(ALL_KEYS, n_days=2500, mode="trend", seed=4)
    base = Config(keys=ALL_KEYS)
    nobuf = Config(keys=ALL_KEYS,
                   risk=RiskConfig(**{**base.risk.__dict__, "buffer_fraction": 0.0}))
    t_on = run_backtest(prices, base).turnover.mean()
    t_off = run_backtest(prices, nobuf).turnover.mean()
    check("buffering reduit le turnover", t_on < t_off * 0.8,
          f"turnover {t_off:.4f} -> {t_on:.4f} ({1 - t_on / t_off:.0%} de moins)")


def test_costs_are_charged():
    """Les couts doivent etre non nuls et mordre le rendement brut."""
    prices = generate(ALL_KEYS, n_days=2000, mode="trend", seed=6)
    res = run_backtest(prices, Config(keys=ALL_KEYS))
    total_cost = res.costs.sum()
    check("couts effectivement preleves", total_cost > 0,
          f"cout cumule = {total_cost:.2%} du capital")


def test_holding_period_is_swing():
    """Contrainte utilisateur : pas d'intraday. Detention mediane > 2 semaines."""
    from quant.backtest import extract_trades
    prices = generate(ALL_KEYS, n_days=3000, mode="trend", seed=8)
    res = run_backtest(prices, Config(keys=ALL_KEYS))
    trades = extract_trades(res.positions, res.pnl_by_asset)
    median_bars = trades.bars.median()
    check("duree de detention swing (> 10 jours)", median_bars > 10,
          f"mediane = {median_bars:.0f} jours")


if __name__ == "__main__":
    print("Validation du moteur\n" + "=" * 60)
    for fn in (test_no_lookahead_shift, test_no_edge_on_random_walk,
               test_vol_targeting_works, test_risk_balance_btc_vs_gold,
               test_buffering_cuts_turnover, test_costs_are_charged,
               test_holding_period_is_swing):
        fn()
    print("=" * 60)
    print("TOUS LES TESTS PASSENT" if not FAILED else f"ECHECS : {FAILED}")
    sys.exit(1 if FAILED else 0)
