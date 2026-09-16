"""Dimensionnement des positions : vol targeting, IDM, buffering.

C'est ici que se joue l'essentiel de la performance ajustee du risque.
Le signal decide de la DIRECTION ; ce module decide de la TAILLE, et la
taille compte davantage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RiskConfig


def instrument_diversification_multiplier(returns: pd.DataFrame,
                                          weights: np.ndarray,
                                          cfg: RiskConfig) -> pd.Series:
    """IDM = 1 / sqrt(w' C w), sur correlation glissante.

    Interpretation directe : de combien on peut lever le book parce que les
    positions ne bougent pas ensemble. IDM ~1.0 = aucune diversification
    (tout est le meme pari). IDM ~2.0 = on peut doubler l'exposition a vol
    constante. C'est le chiffre qui quantifie le cout de ne trader que 2 actifs.
    """
    idm = pd.Series(np.nan, index=returns.index, dtype=float)
    window = cfg.corr_window
    # recalcul mensuel : la matrice de correlation bouge lentement et
    # la recalculer chaque jour coute cher pour rien
    for i in range(window, len(returns), 21):
        block = returns.iloc[i - window:i]
        valid = block.columns[block.notna().sum() >= window // 2]
        if len(valid) == 0:
            continue
        corr = block[valid].corr().to_numpy()
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        w = weights[[returns.columns.get_loc(c) for c in valid]]
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        variance = float(w @ corr @ w)
        idm.iloc[i] = 1.0 / np.sqrt(max(variance, 1e-6))
    return idm.ffill().clip(1.0, cfg.max_idm).bfill()


def target_notionals(forecasts: pd.DataFrame, vols: pd.DataFrame,
                     returns: pd.DataFrame, capital: float,
                     forecast_target: float, cfg: RiskConfig
                     ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Notionnel cible par instrument, en devise de compte.

    notionnel_i = (forecast_i / target) * IDM * poids_i * capital
                  * vol_cible / vol_annualisee_i

    La division par la vol de l'instrument est le coeur du vol targeting :
    chaque position porte le meme risque en $, que ce soit BTC ou l'or.
    """
    keys = list(forecasts.columns)
    n = len(keys)
    # poids egaux : sans vue ex-ante fiable sur les Sharpe relatifs, le
    # 1/N est difficile a battre hors-echantillon (Kelly estime = bruit).
    weights = np.full(n, 1.0 / n)

    idm = instrument_diversification_multiplier(returns, weights, cfg)
    ann_vol = vols * np.sqrt(cfg.trading_days)

    scaled_forecast = forecasts / forecast_target
    risk_budget = capital * cfg.vol_target * weights  # $ de vol par instrument
    notional = scaled_forecast.mul(risk_budget, axis=1).div(ann_vol)
    notional = notional.mul(idm, axis=0)

    # plafond par instrument : empeche BTC de squatter le budget de risque
    # quand sa vol s'effondre temporairement
    cap = capital * cfg.max_instrument_leverage * weights
    notional = notional.clip(lower=-cap, upper=cap, axis=1)

    # plafond de levier brut portefeuille (garde-fou de dernier recours)
    gross = notional.abs().sum(axis=1)
    overshoot = (gross / (capital * cfg.max_gross_leverage)).clip(lower=1.0)
    notional = notional.div(overshoot, axis=0)

    # notionnel "unitaire" : la taille que porterait un forecast pleine
    # echelle. Sert d'echelle de reference au buffering, par instrument.
    unit = pd.DataFrame(
        np.ones((len(notional), len(keys))), index=notional.index, columns=keys
    ).mul(risk_budget, axis=1).div(ann_vol).mul(idm, axis=0)
    unit = unit.clip(upper=cap, axis=1).ffill().fillna(0.0)

    return notional.fillna(0.0), idm, unit


def apply_buffering(target: pd.DataFrame, unit: pd.DataFrame,
                    cfg: RiskConfig) -> pd.DataFrame:
    """Ne rebalance que si l'ecart depasse une bande morte.

    La bande est proportionnelle au notionnel unitaire de CHAQUE instrument.
    Une bande globale en dollars serait absurde : la position naturelle sur
    du 10 ans US (vol 6%) est ~8x celle de BTC (vol 50%) a risque egal.

    Le trend-following n'a aucun besoin d'etre precis a 1% pres sur la taille.
    Cette bande divise le turnover par ~2-3 pour une degradation de signal
    negligeable : c'est le meilleur ratio cout/benefice du systeme.
    """
    tgt = target.to_numpy(dtype=float)
    bands = (cfg.buffer_fraction * unit.reindex_like(target).to_numpy(dtype=float))
    bands = np.maximum(np.nan_to_num(bands), 1e-9)

    held = np.zeros_like(tgt)
    current = np.zeros(tgt.shape[1])
    for t in range(tgt.shape[0]):
        band = bands[t]
        gap = tgt[t] - current
        move = np.abs(gap) > band
        # on se deplace jusqu'au bord de la bande, pas jusqu'a la cible :
        # evite de re-trader immediatement au moindre bruit
        current = np.where(move, tgt[t] - np.sign(gap) * band * 0.5, current)
        held[t] = current
    return pd.DataFrame(held, index=target.index, columns=target.columns)
