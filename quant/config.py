"""Configuration du systeme : univers, couts, parametres de risque.

Aucun parametre n'est optimise sur les donnees. Les valeurs viennent de la
litterature (Carver, AQR) et des couts reels broker. C'est volontaire :
l'over-fitting sur 2 actifs est le premier risque de ce genre de projet.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Instrument:
    key: str
    yahoo: str            # ticker Yahoo Finance
    binance: str | None   # symbole perp Binance (funding dispo si non None)
    asset_class: str
    cost_bps: float       # cout aller-retour complet : spread + commission + slippage
    core: bool = False    # True = actif que l'utilisateur trade deja


# --- Univers -----------------------------------------------------------------
# Coeur : BTC + XAU (demande utilisateur).
# Satellites : ajoutes pour la diversification. Meme brokers, tres liquides,
# couts faibles. Chaque classe d'actif apporte un driver macro distinct.
UNIVERSE: list[Instrument] = [
    # --- coeur ---
    Instrument("BTC",  "BTC-USD",    "BTCUSDT", "crypto", cost_bps=6.0,  core=True),
    Instrument("XAU",  "GC=F",       None,      "metal",  cost_bps=3.0,  core=True),
    # --- satellites (diversification) ---
    Instrument("ETH",  "ETH-USD",    "ETHUSDT", "crypto", cost_bps=8.0),
    Instrument("XAG",  "SI=F",       None,      "metal",  cost_bps=8.0),
    Instrument("SPX",  "ES=F",       None,      "equity", cost_bps=1.5),
    Instrument("WTI",  "CL=F",       None,      "energy", cost_bps=4.0),
    Instrument("EUR",  "EURUSD=X",   None,      "fx",     cost_bps=1.0),
    Instrument("UST",  "ZN=F",       None,      "rates",  cost_bps=1.5),
]

CORE_KEYS = [i.key for i in UNIVERSE if i.core]
ALL_KEYS = [i.key for i in UNIVERSE]
BY_KEY = {i.key: i for i in UNIVERSE}

# Series macro utilisees par les overlays (pas tradees)
MACRO_TICKERS = {"DXY": "DX-Y.NYB", "UST10Y": "^TNX"}


@dataclass(frozen=True)
class RiskConfig:
    # Vol cible annualisee au niveau portefeuille.
    # 12% = agressif mais tenable. Au-dela, le DD devient psychologiquement
    # intenable et le risque de ruine sur queue crypto explose.
    vol_target: float = 0.12

    # Estimation de vol : melange court terme / long terme.
    # Le pur EWMA sous-estime la vol juste avant les chocs -> on l'ancre.
    vol_ewma_span: int = 32
    vol_long_window: int = 512
    vol_long_weight: float = 0.30
    trading_days: int = 256

    # Plafond d'exposition par instrument, en multiple de sa part cible.
    # Empeche un seul actif (BTC) de manger le budget de risque.
    max_instrument_leverage: float = 2.0

    # Plafond de levier brut portefeuille. Garde-fou de dernier recours.
    max_gross_leverage: float = 4.0

    # Buffering : on ne rebalance que si l'ecart depasse ce % de la position
    # cible moyenne. Divise le turnover par ~3 sans degrader le signal.
    buffer_fraction: float = 0.10

    # Multiplicateur de diversification (IDM), borne haute.
    # Sans borne, une matrice de correlation instable peut sur-lever.
    max_idm: float = 2.5
    corr_window: int = 256


@dataclass(frozen=True)
class SignalConfig:
    # Paires EWMA (rapide, lent). Ensemble multi-horizon : aucun lookback
    # unique n'est robuste, la moyenne l'est. Horizons longs uniquement
    # -> duree de detention de plusieurs semaines, zero intraday.
    ewmac_pairs: tuple[tuple[int, int], ...] = ((16, 64), (32, 128), (64, 256))
    # Fenetres de breakout Donchian, en jours.
    breakout_windows: tuple[int, ...] = (60, 120, 240)
    # Cap du forecast (echelle Carver : cible 10, cap 20 -> position x2 max)
    forecast_target: float = 10.0
    forecast_cap: float = 20.0
    # Poids trend-continu vs breakout dans l'ensemble
    ewmac_weight: float = 0.5


@dataclass(frozen=True)
class OverlayConfig:
    # BTC : funding perp. Funding tres positif = longs surcharges en levier,
    # risque de cascade de liquidations -> on allege, on n'inverse pas.
    use_funding: bool = True
    funding_hot_annual: float = 0.30   # >30%/an de funding = crowded
    funding_haircut: float = 0.50      # on coupe la moitie du long

    # XAU : taux reels / DXY. L'or n'a pas de carry, il a un cout d'opportunite.
    use_gold_macro: bool = True
    dxy_window: int = 120

    # Blackout news : pas de NOUVELLE entree dans cette fenetre.
    # Sur donnees daily c'est approximatif ; utile surtout en live.
    use_news_blackout: bool = False


@dataclass
class Config:
    risk: RiskConfig = field(default_factory=RiskConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    capital: float = 100_000.0
    keys: list[str] = field(default_factory=lambda: list(ALL_KEYS))
