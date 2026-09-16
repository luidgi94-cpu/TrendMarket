#!/usr/bin/env python3
"""Lance le backtest et compare les univers.

Usage:
    python run_backtest.py                 # donnees reelles (reseau requis)
    python run_backtest.py --synthetic     # validation du moteur, hors-ligne
    python run_backtest.py --refresh       # force le retelechargement
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from quant import metrics
from quant.backtest import extract_trades, run_backtest
from quant.config import ALL_KEYS, CORE_KEYS, Config
from quant.synthetic import generate, synthetic_funding

pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")


def evaluate(name: str, prices, keys, funding=None, macro=None, verbose=True):
    cfg = Config(keys=list(keys))
    res = run_backtest(prices, cfg, funding=funding, macro=macro)
    trades = extract_trades(res.positions, res.pnl_by_asset)

    stats = metrics.summary(res.returns, res.equity, res.gross_returns,
                            res.costs, res.turnover)
    tstats = metrics.trade_stats(trades)

    if verbose:
        print(f"\n{'=' * 74}\n{name}  ({len(keys)} instruments : {', '.join(keys)})\n{'=' * 74}")
        print("-- Performance ----------------------------------------------")
        print(metrics.fmt(stats))
        print("-- Trades ---------------------------------------------------")
        print(metrics.fmt(tstats))
        print(f"-- Diversification (IDM) ------------------------------------")
        print(f"  IDM moyen              {res.idm.mean():>8.2f}   "
              f"(1.0 = aucune diversification)")
        print("-- Contribution par actif -----------------------------------")
        print(metrics.per_asset(res.pnl_by_asset, cfg.capital).to_string())
    return res, stats, tstats


def oos_split(res, label: str) -> None:
    """Coupe l'echantillon en deux. Les parametres n'etant pas optimises,
    un ecart de Sharpe important entre les deux moities signale une
    dependance au regime, pas un over-fitting."""
    r = res.returns.dropna()
    cut = len(r) // 2
    print(f"\n-- Stabilite temporelle : {label} --------------------------")
    for part, seg in (("1ere moitie", r.iloc[:cut]), ("2eme moitie", r.iloc[cut:])):
        eq = (1 + seg).cumprod()
        s = metrics.summary(seg, eq)
        print(f"  {part}: Sharpe {s['Sharpe']:>5.2f} | CAGR {s['CAGR']:>6.1%} "
              f"| maxDD {s['max_DD']:>6.1%} | winrate_j {s['winrate_jours']:>5.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--mc", type=int, default=0,
                    metavar="N", help="Monte Carlo sur N chemins synthetiques")
    args = ap.parse_args()

    if args.mc:
        from quant.montecarlo import paired_test, report, run_paths
        universes = {"A_coeur_2": CORE_KEYS,
                     "B_plus_metaux": CORE_KEYS + ["ETH", "XAG"],
                     "C_diversifie_8": ALL_KEYS}
        print(f"Monte Carlo : {args.mc} chemins synthetiques independants")
        print("Sharpe plafond d'un oracle parfait sur ce monde : 0.40\n")
        df = run_paths(universes, ALL_KEYS, n_paths=args.mc, n_days=3000)
        print(report(df).round(3).to_string())
        print("\n=== Test apparie : BTC+XAU  ->  8 instruments ===")
        for k, v in paired_test(df, "A_coeur_2", "C_diversifie_8").items():
            print(f"  {k:<26} {v:>8.3f}")
        return 0

    if args.synthetic:
        print("### DONNEES SYNTHETIQUES - validation du moteur, PAS d'un edge ###")
        prices = generate(ALL_KEYS, n_days=4000, mode="trend")
        funding = synthetic_funding(prices, ["BTC", "ETH"])
        macro = None
    else:
        from quant.data import load_funding, load_macro, load_prices
        print("Telechargement des donnees...")
        prices = load_prices(ALL_KEYS, start=args.start, refresh=args.refresh)
        funding = load_funding(ALL_KEYS, refresh=args.refresh)
        macro = load_macro(refresh=args.refresh)
        print(f"  {len(prices)} jours | {list(prices.columns)}")

    available = [k for k in prices.columns]
    core = [k for k in CORE_KEYS if k in available]

    res_core, s_core, t_core = evaluate("A. COEUR SEUL (BTC + XAU)",
                                        prices, core, funding, macro)
    res_div, s_div, t_div = evaluate("B. UNIVERS DIVERSIFIE",
                                      prices, available, funding, macro)

    oos_split(res_core, "coeur")
    oos_split(res_div, "diversifie")

    print(f"\n{'=' * 74}\nVERDICT : cout de la contrainte '2 actifs'\n{'=' * 74}")
    rows = {
        "Sharpe": (s_core["Sharpe"], s_div["Sharpe"]),
        "CAGR": (s_core["CAGR"], s_div["CAGR"]),
        "max drawdown": (s_core["max_DD"], s_div["max_DD"]),
        "mois sous l'eau": (s_core["mois_sous_leau"], s_div["mois_sous_leau"]),
        "winrate trades": (t_core["winrate_trades"], t_div["winrate_trades"]),
        "esperance (R)": (t_core["esperance_en_R"], t_div["esperance_en_R"]),
        "duree med. (j)": (t_core["duree_mediane_jours"], t_div["duree_mediane_jours"]),
        "IDM": (res_core.idm.mean(), res_div.idm.mean()),
    }
    print(f"  {'metrique':<18}{'BTC+XAU':>12}{'diversifie':>14}")
    for k, (a, b) in rows.items():
        fa = f"{a:.1%}" if "rate" in k or "CAGR" in k or "drawdown" in k else f"{a:.2f}"
        fb = f"{b:.1%}" if "rate" in k or "CAGR" in k or "drawdown" in k else f"{b:.2f}"
        print(f"  {k:<18}{fa:>12}{fb:>14}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
