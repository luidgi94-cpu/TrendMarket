"""Types de base : bougie, evenement de structure, zone."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List


@dataclass
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    spread: float = 0.0   # en points, si le flux le fournit (MT5)

    @property
    def bullish(self) -> bool:
        return self.close > self.open

    @property
    def bearish(self) -> bool:
        return self.close < self.open

    @property
    def body_high(self) -> float:
        return max(self.open, self.close)

    @property
    def body_low(self) -> float:
        return min(self.open, self.close)


@dataclass
class StructureEvent:
    """BOS ou CHoCH. `idx` est la bougie dont la CLOTURE casse le niveau."""
    kind: str          # "BOS" | "CHoCH"
    direction: str     # "bull" | "bear"
    level: float       # niveau casse
    idx: int           # bougie de cassure (= instant de connaissance)
    ts: datetime
    leg_start_idx: Optional[int] = None   # debut de la jambe impulsive
    leg_end_idx: Optional[int] = None


@dataclass
class Zone:
    """Zone de prix horodatee. `confirmed_idx` = bougie a partir de laquelle
    la zone est CONNAISSABLE. Toute lecture anterieure serait du look-ahead."""
    kind: str          # "OB" | "FVG" | "VG"
    direction: str     # "bull" | "bear"
    low: float
    high: float
    origin_idx: int    # bougie qui forme la zone
    confirmed_idx: int
    ts: datetime
    tf: str = ""
    mitigated_idx: Optional[int] = None
    meta: dict = field(default_factory=dict)

    @property
    def mitigated(self) -> bool:
        return self.mitigated_idx is not None

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2.0

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    def overlaps(self, lo: float, hi: float) -> bool:
        return not (hi < self.low or lo > self.high)


@dataclass
class Liquidity:
    """Bassin de liquidite d'une session."""
    session: str
    side: str          # "buy" (sommet) | "sell" (creux)
    level: float
    day: str
    formed_idx: int
    swept_idx: Optional[int] = None

    @property
    def swept(self) -> bool:
        return self.swept_idx is not None
