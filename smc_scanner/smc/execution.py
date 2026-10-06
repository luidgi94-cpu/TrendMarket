"""Protocole de confirmation : bascule M15 -> M5 -> gachette M1."""
from typing import List, Optional, Dict
from .model import Candle, Zone


def is_engulfing(prev: Candle, cur: Candle, direction: str) -> bool:
    """Englobante : le corps courant couvre entierement le corps precedent,
    et cloture dans le sens voulu."""
    covers = cur.body_low <= prev.body_low and cur.body_high >= prev.body_high
    if not covers or cur.body_high == cur.body_low:
        return False
    return cur.bullish if direction == "bull" else cur.bearish


def is_inside_bar(prev: Candle, cur: Candle) -> bool:
    """Inside bar : la bougie courante est contenue dans la precedente (meches)."""
    return cur.high <= prev.high and cur.low >= prev.low


def is_overlap(prev: Candle, cur: Candle, direction: str) -> bool:
    """Chevauchante (lap) : la bougie empiete sur le corps precedent sans
    l'englober, en cloturant dans le sens voulu."""
    if is_engulfing(prev, cur, direction) or is_inside_bar(prev, cur):
        return False
    if direction == "bull":
        return cur.low < prev.body_low <= cur.close and cur.bullish
    return cur.high > prev.body_high >= cur.close and cur.bearish


def m1_trigger(m1: List[Candle], direction: str, start_idx: int = 1,
               patterns=("engulfing", "inside", "overlap")) -> Optional[Dict]:
    """Premier pattern M1 valide a partir de start_idx."""
    for i in range(max(1, start_idx), len(m1)):
        prev, cur = m1[i - 1], m1[i]
        if "engulfing" in patterns and is_engulfing(prev, cur, direction):
            return {"idx": i, "pattern": "Englobante", "candle": cur}
        if "inside" in patterns and is_inside_bar(prev, cur):
            return {"idx": i, "pattern": "Chevauchante (Inside Bar)", "candle": cur}
        if "overlap" in patterns and is_overlap(prev, cur, direction):
            return {"idx": i, "pattern": "Chevauchante (Lap)", "candle": cur}
    return None


def approach_state(zone: Zone, last: Candle, atr_ref: float) -> str:
    """Ou en est le prix par rapport a la zone."""
    if zone.overlaps(last.low, last.high):
        return "DANS LA ZONE"
    dist = zone.low - last.close if last.close < zone.low else last.close - zone.high
    if atr_ref > 0 and dist <= 1.0 * atr_ref:
        return "APPROCHE (< 1 ATR)"
    if atr_ref > 0 and dist <= 3.0 * atr_ref:
        return "SURVEILLANCE (< 3 ATR)"
    return "ELOIGNE"
