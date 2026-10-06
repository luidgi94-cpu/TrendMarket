"""Order Blocks, FVG, Volume Gaps, OTE, mitigation."""
from typing import List, Optional
from .model import Candle, Zone, StructureEvent
from .data import atr


def find_order_blocks(candles: List[Candle], events: List[StructureEvent],
                      tf: str = "", lookback: int = 30,
                      min_impulse_atr: float = 1.5,
                      use_body: bool = True) -> List[Zone]:
    """Derniere bougie opposee avant l'impulsion qui casse la structure.

    L'OB est date a `confirmed_idx = event.idx` : on ne peut PAS savoir
    qu'une bougie est un OB avant que la cassure ne soit cloturee.
    C'est le point que la plupart des scanners ratent, et la raison pour
    laquelle ils paraissent excellents en visuel et echouent en live.
    """
    a = atr(candles)
    out: List[Zone] = []
    seen: set = set()   # une bougie ne produit qu'un seul OB, meme si
    for ev in events:   # plusieurs evenements la confirment (BOS puis CHoCH)
        bi = ev.idx
        want_bearish_candle = ev.direction == "bull"
        start = max(0, bi - lookback)
        ob_idx: Optional[int] = None
        for j in range(bi - 1, start - 1, -1):
            c = candles[j]
            if (c.bearish if want_bearish_candle else c.bullish):
                ob_idx = j
                break
        if ob_idx is None:
            continue

        if (ob_idx, ev.direction) in seen:
            continue
        seen.add((ob_idx, ev.direction))

        ob = candles[ob_idx]
        # filtre d'impulsion : le deplacement OB -> cassure doit etre significatif
        ref = a[bi] if a[bi] > 0 else 1e-9
        disp = (max(x.high for x in candles[ob_idx:bi + 1]) - ob.body_low
                if ev.direction == "bull"
                else ob.body_high - min(x.low for x in candles[ob_idx:bi + 1]))
        if disp < min_impulse_atr * ref:
            continue

        lo = ob.body_low if use_body else ob.low
        hi = ob.body_high if use_body else ob.high
        out.append(Zone("OB", ev.direction, lo, hi, ob_idx, bi, ob.ts, tf,
                        meta={"event": ev.kind, "break_level": ev.level,
                              "impulse_atr": round(disp / ref, 2),
                              "leg_start_idx": ev.leg_start_idx,
                              "leg_end_idx": ev.leg_end_idx}))
    mark_mitigation(candles, out)
    return out


def find_fvg(candles: List[Candle], tf: str = "", min_size: float = 0.0) -> List[Zone]:
    """Fair Value Gap : sur 3 bougies, les meches 1 et 3 ne se croisent pas.
    Connu des la cloture de la 3e bougie."""
    out: List[Zone] = []
    for i in range(2, len(candles)):
        c1, c3 = candles[i - 2], candles[i]
        if c3.low > c1.high and (c3.low - c1.high) >= min_size:
            out.append(Zone("FVG", "bull", c1.high, c3.low, i - 1, i, candles[i - 1].ts, tf))
        elif c3.high < c1.low and (c1.low - c3.high) >= min_size:
            out.append(Zone("FVG", "bear", c3.high, c1.low, i - 1, i, candles[i - 1].ts, tf))
    mark_mitigation(candles, out)
    return out


def find_volume_gaps(candles: List[Candle], tf: str = "") -> List[Zone]:
    """Trou de cotation reel : aucun echange entre deux bougies
    (ni corps ni meche ne se recouvrent)."""
    out: List[Zone] = []
    for i in range(1, len(candles)):
        p, c = candles[i - 1], candles[i]
        if c.low > p.high:
            out.append(Zone("VG", "bull", p.high, c.low, i, i, c.ts, tf))
        elif c.high < p.low:
            out.append(Zone("VG", "bear", c.high, p.low, i, i, c.ts, tf))
    mark_mitigation(candles, out)
    return out


def mark_mitigation(candles: List[Candle], zones: List[Zone]) -> None:
    """Une zone est mitigee des que le prix y revient APRES sa confirmation."""
    for z in zones:
        if z.mitigated_idx is not None:
            continue
        for i in range(z.confirmed_idx + 1, len(candles)):
            c = candles[i]
            if z.overlaps(c.low, c.high):
                z.mitigated_idx = i
                break


def ote_zone(leg_low: float, leg_high: float, direction: str) -> tuple:
    """Optimal Trade Entry : retracement strict 0.618 - 0.786.
    Trace du bas vers le haut sur une jambe acheteuse, inverse sur une vendeuse.
    Retourne (borne_basse, borne_haute)."""
    rng = leg_high - leg_low
    if rng <= 0:
        return (0.0, 0.0)
    if direction == "bull":
        return (leg_high - 0.786 * rng, leg_high - 0.618 * rng)
    return (leg_low + 0.618 * rng, leg_low + 0.786 * rng)


def in_ote(zone: Zone, candles: List[Candle]) -> bool:
    """La zone chevauche-t-elle l'OTE de sa propre jambe impulsive ?"""
    s, e = zone.meta.get("leg_start_idx"), zone.meta.get("leg_end_idx")
    if s is None or e is None or s >= e or e >= len(candles):
        return False
    leg = candles[s:e + 1]
    lo, hi = min(c.low for c in leg), max(c.high for c in leg)
    a, b = ote_zone(lo, hi, zone.direction)
    return b > a and zone.overlaps(a, b)


def adjacent_fvg(zone: Zone, fvgs: List[Zone], tol: float) -> Optional[Zone]:
    """FVG de meme sens, immediatement au-dessus (bull) / au-dessous (bear),
    et connu au plus tard a la confirmation de la zone."""
    best, best_gap = None, None
    for f in fvgs:
        if f.direction != zone.direction or f.confirmed_idx > zone.confirmed_idx:
            continue
        gap = f.low - zone.high if zone.direction == "bull" else zone.low - f.high
        if -tol <= gap <= tol and (best_gap is None or abs(gap) < abs(best_gap)):
            best, best_gap = f, gap
    return best
