"""Comparaison Monte Carlo d'univers.

Un backtest = un chemin = une anecdote. Sur 2 instruments, l'ecart-type du
Sharpe estime sur 15 ans est de l'ordre de +/-0.25 : comparer deux univers
sur un seul chemin ne prouve strictement rien.

On regenere donc N marches independants et on compare les DISTRIBUTIONS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics
from .backtest import extract_trades, run_backtest
from .config import Config
from .synthetic import generate


def run_paths(universes: dict[str, list[str]], all_keys: list[str],
              n_paths: int = 30, n_days: int = 3000,
              mode: str = "trend") -> pd.DataFrame:
    rows = []
    for seed in range(n_paths):
        prices = generate(all_keys, n_days=n_days, mode=mode, seed=1000 + seed)
        for name, keys in universes.items():
            cfg = Config(keys=list(keys))
            res = run_backtest(prices, cfg)
            s = metrics.summary(res.returns, res.equity)
            t = metrics.trade_stats(extract_trades(res.positions, res.pnl_by_asset))
            rows.append({
                "univers": name, "seed": seed,
                "Sharpe": s["Sharpe"], "CAGR": s["CAGR"],
                "max_DD": s["max_DD"], "vol": s["vol_ann"],
                "mois_sous_leau": s["mois_sous_leau"],
                "skew_mensuel": s["skew_mensuel"],
                "winrate_trades": t.get("winrate_trades", np.nan),
                "duree_med": t.get("duree_mediane_jours", np.nan),
                "IDM": res.idm.mean(),
            })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("univers").agg(
        Sharpe_median=("Sharpe", "median"),
        Sharpe_q25=("Sharpe", lambda x: x.quantile(0.25)),
        Sharpe_q75=("Sharpe", lambda x: x.quantile(0.75)),
        prob_Sharpe_neg=("Sharpe", lambda x: (x < 0).mean()),
        maxDD_median=("max_DD", "median"),
        maxDD_pire=("max_DD", "min"),
        mois_sous_leau_med=("mois_sous_leau", "median"),
        winrate_median=("winrate_trades", "median"),
        duree_med=("duree_med", "median"),
        IDM=("IDM", "mean"),
    )
    return agg


def paired_test(df: pd.DataFrame, a: str, b: str) -> dict:
    """Test apparie : meme chemin de marche, deux univers.

    L'appariement elimine le bruit du chemin et isole l'effet de l'univers.
    """
    from scipy import stats
    pa = df[df.univers == a].set_index("seed")
    pb = df[df.univers == b].set_index("seed")
    common = pa.index.intersection(pb.index)
    d_sharpe = (pb.loc[common, "Sharpe"] - pa.loc[common, "Sharpe"]).to_numpy()
    d_dd = (pb.loc[common, "max_DD"] - pa.loc[common, "max_DD"]).to_numpy()
    t_s, p_s = stats.ttest_1samp(d_sharpe, 0.0)
    t_d, p_d = stats.ttest_1samp(d_dd, 0.0)
    return {
        "n_chemins": len(common),
        "delta_Sharpe_moyen": float(d_sharpe.mean()),
        "p_value_Sharpe": float(p_s),
        "chemins_ou_B_gagne": float((d_sharpe > 0).mean()),
        "delta_maxDD_moyen": float(d_dd.mean()),
        "p_value_maxDD": float(p_d),
        "chemins_ou_B_moins_DD": float((d_dd > 0).mean()),
    }
