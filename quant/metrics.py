"""Statistiques de performance.

On rapporte le winrate parce qu'il est demande, mais il arrive apres le
Sharpe, le drawdown et le skew dans l'ordre d'importance. Un systeme se
juge sur son rendement par unite de risque et sur sa capacite a survivre
a ses propres queues, pas sur sa frequence de gain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 256


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def summary(returns: pd.Series, equity: pd.Series,
            gross_returns: pd.Series | None = None,
            costs: pd.Series | None = None,
            turnover: pd.Series | None = None) -> dict:
    r = returns.dropna()
    if len(r) < 2:
        return {}

    years = len(r) / TRADING_DAYS
    ann_ret = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = ann_ret / downside if downside > 0 else np.nan

    dd = drawdown(equity)
    max_dd = dd.min()

    # plus longue periode sous l'eau, en mois : le vrai test psychologique
    under = (dd < -1e-9).astype(int)
    longest, run = 0, 0
    for v in under:
        run = run + 1 if v else 0
        longest = max(longest, run)

    out = {
        "annees": round(years, 1),
        "CAGR": ann_ret,
        "vol_ann": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "max_DD": max_dd,
        "Calmar": ann_ret / abs(max_dd) if max_dd < 0 else np.nan,
        "skew_quotidien": stats.skew(r.to_numpy()),
        "skew_mensuel": stats.skew(
            r.resample("ME").apply(lambda x: (1 + x).prod() - 1).dropna().to_numpy()
        ) if len(r) > 60 else np.nan,
        "winrate_jours": (r > 0).mean(),
        "pire_12m": r.rolling(TRADING_DAYS).apply(
            lambda x: (1 + x).prod() - 1, raw=True).min(),
        "mois_sous_leau": round(longest / 21.0, 1),
    }

    if gross_returns is not None and costs is not None:
        gross = gross_returns.dropna()
        gross_ann = gross.mean() * TRADING_DAYS
        cost_ann = costs.dropna().mean() * TRADING_DAYS
        out["rendement_brut_ann"] = gross_ann
        out["couts_ann"] = cost_ann
        out["couts_%_du_brut"] = cost_ann / gross_ann if gross_ann > 0 else np.nan
    if turnover is not None:
        out["turnover_ann"] = turnover.dropna().mean() * TRADING_DAYS

    return out


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    wins = trades[trades.pnl > 0]
    losses = trades[trades.pnl <= 0]
    avg_win = wins.pnl.mean() if len(wins) else 0.0
    avg_loss = abs(losses.pnl.mean()) if len(losses) else 0.0
    winrate = len(wins) / len(trades)
    return {
        "nb_trades": len(trades),
        "winrate_trades": winrate,
        "gain_moyen": avg_win,
        "perte_moyenne": avg_loss,
        "ratio_gain_perte": avg_win / avg_loss if avg_loss > 0 else np.nan,
        # esperance en multiples de la perte moyenne : la seule metrique
        # qui combine winrate et payoff. C'est elle qui doit etre > 0.
        "esperance_en_R": (winrate * avg_win - (1 - winrate) * avg_loss) / avg_loss
                          if avg_loss > 0 else np.nan,
        "duree_mediane_jours": float(trades.bars.median()),
        "duree_moyenne_jours": float(trades.bars.mean()),
        "part_pnl_top5": (trades.nlargest(min(5, len(trades)), "pnl").pnl.sum()
                          / trades.pnl.sum()) if trades.pnl.sum() > 0 else np.nan,
    }


def per_asset(pnl_by_asset: pd.DataFrame, capital: float) -> pd.DataFrame:
    rows = {}
    total = pnl_by_asset.sum().sum()
    for key in pnl_by_asset.columns:
        series = pnl_by_asset[key] / capital
        ann_vol = series.std() * np.sqrt(TRADING_DAYS)
        ann_ret = series.mean() * TRADING_DAYS
        rows[key] = {
            "contrib_PnL_%": pnl_by_asset[key].sum() / total if total != 0 else np.nan,
            "rendement_ann": ann_ret,
            "vol_ann": ann_vol,
            "Sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        }
    return pd.DataFrame(rows).T


def fmt(d: dict) -> str:
    pct = {"CAGR", "vol_ann", "max_DD", "winrate_jours", "pire_12m",
           "rendement_brut_ann", "couts_ann", "couts_%_du_brut",
           "winrate_trades", "contrib_PnL_%", "rendement_ann"}
    lines = []
    for k, v in d.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            lines.append(f"  {k:<22} n/a")
        elif k in pct:
            lines.append(f"  {k:<22} {v:>8.1%}")
        elif isinstance(v, float):
            lines.append(f"  {k:<22} {v:>8.2f}")
        else:
            lines.append(f"  {k:<22} {v:>8}")
    return "\n".join(lines)
