"""Pivots, BOS, CHoCH, momentum.

Regle anti-look-ahead : un pivot forme en i n'est CONNU qu'en i+right.
Toute la machine de structure respecte cet horodatage de connaissance.
"""
from typing import List, Optional, Tuple
from .model import Candle, StructureEvent


class Pivot:
    __slots__ = ("idx", "price", "side", "known_idx")

    def __init__(self, idx: int, price: float, side: str, known_idx: int):
        self.idx = idx
        self.price = price
        self.side = side          # "high" | "low"
        self.known_idx = known_idx

    def __repr__(self):
        return f"Pivot({self.side} {self.price:.2f} @{self.idx} known@{self.known_idx})"


def find_pivots(candles: List[Candle], left: int = 2, right: int = 2) -> List[Pivot]:
    """Fractales a `left`/`right` bougies. Le pivot est date a sa confirmation."""
    out: List[Pivot] = []
    for i in range(left, len(candles) - right):
        win = candles[i - left:i + right + 1]
        c = candles[i]
        if c.high == max(x.high for x in win) and \
           all(c.high > candles[j].high for j in range(i - left, i)):
            out.append(Pivot(i, c.high, "high", i + right))
        if c.low == min(x.low for x in win) and \
           all(c.low < candles[j].low for j in range(i - left, i)):
            out.append(Pivot(i, c.low, "low", i + right))
    out.sort(key=lambda p: (p.known_idx, p.idx))
    return out


def build_structure(candles: List[Candle], left: int = 2, right: int = 2
                    ) -> Tuple[List[StructureEvent], List[Pivot]]:
    """Derive la sequence BOS/CHoCH.

    BOS  : cloture au-dela du dernier sommet (haussier) / creux (baissier)
           majeur, DANS le sens de la tendance -> continuation.
    CHoCH: cloture sous le dernier creux ayant entraine le plus haut
           (en tendance haussiere), et symetriquement. -> retournement.
    """
    pivots = find_pivots(candles, left, right)
    by_known: dict = {}
    for p in pivots:
        by_known.setdefault(p.known_idx, []).append(p)

    events: List[StructureEvent] = []
    trend: Optional[str] = None
    last_high: Optional[Pivot] = None   # sommet de reference (cible BOS haussier)
    last_low: Optional[Pivot] = None    # creux de reference (cible BOS baissier)
    choch_low: Optional[Pivot] = None   # creux ayant entraine le dernier sommet
    choch_high: Optional[Pivot] = None  # sommet ayant entraine le dernier creux
    prev_low: Optional[Pivot] = None
    prev_high: Optional[Pivot] = None

    for i, c in enumerate(candles):
        # 1) integrer les pivots devenus connus a cette bougie
        for p in by_known.get(i, []):
            if p.side == "high":
                choch_low = prev_low          # le creux qui a mene a ce sommet
                last_high = p
                prev_high = p
            else:
                choch_high = prev_high
                last_low = p
                prev_low = p

        # 2) tester les cassures sur la cloture de la bougie courante
        if trend in (None, "bull") and last_high and c.close > last_high.price:
            events.append(StructureEvent(
                "BOS", "bull", last_high.price, i, c.ts,
                leg_start_idx=(choch_low.idx if choch_low else None),
                leg_end_idx=i))
            trend = "bull"
            last_high = None
        elif trend == "bull" and choch_low and c.close < choch_low.price:
            events.append(StructureEvent(
                "CHoCH", "bear", choch_low.price, i, c.ts,
                leg_start_idx=(prev_high.idx if prev_high else None),
                leg_end_idx=i))
            trend = "bear"
            choch_low = None

        elif trend in (None, "bear") and last_low and c.close < last_low.price:
            events.append(StructureEvent(
                "BOS", "bear", last_low.price, i, c.ts,
                leg_start_idx=(choch_high.idx if choch_high else None),
                leg_end_idx=i))
            trend = "bear"
            last_low = None
        elif trend == "bear" and choch_high and c.close > choch_high.price:
            events.append(StructureEvent(
                "CHoCH", "bull", choch_high.price, i, c.ts,
                leg_start_idx=(prev_low.idx if prev_low else None),
                leg_end_idx=i))
            trend = "bull"
            choch_high = None

    return events, pivots


def momentum(events: List[StructureEvent], upto_idx: Optional[int] = None) -> str:
    """Direction courante d'un TF, lue sur le dernier evenement connu."""
    seq = [e for e in events if upto_idx is None or e.idx <= upto_idx]
    if not seq:
        return "Neutre"
    return "Haussier" if seq[-1].direction == "bull" else "Baissier"
