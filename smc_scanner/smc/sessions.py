"""Hauts/bas de sessions = bassins de liquidite (buy-side / sell-side)."""
from datetime import timezone
from typing import List, Dict
from .model import Candle, Liquidity

# Fenetres en heures UTC [debut, fin). Ajustables selon ton broker.
SESSIONS = {
    "Asie":    (0, 8),
    "Londres": (7, 16),
    "NY":      (12, 21),
}


def session_levels(candles: List[Candle], sessions: Dict[str, tuple] = None
                   ) -> List[Liquidity]:
    """Haut et bas de chaque session, par jour UTC.

    Le niveau n'est connu qu'a la FERMETURE de la session : `formed_idx`
    pointe la derniere bougie de la session, pas celle de l'extreme.
    """
    sessions = sessions or SESSIONS
    acc: Dict[tuple, dict] = {}
    for i, c in enumerate(candles):
        ts = c.ts.astimezone(timezone.utc)
        day = ts.strftime("%Y-%m-%d")
        for name, (h0, h1) in sessions.items():
            if h0 <= ts.hour < h1:
                k = (day, name)
                st = acc.setdefault(k, {"hi": c.high, "lo": c.low, "last": i})
                st["hi"] = max(st["hi"], c.high)
                st["lo"] = min(st["lo"], c.low)
                st["last"] = i

    out: List[Liquidity] = []
    for (day, name), st in sorted(acc.items()):
        out.append(Liquidity(name, "buy", st["hi"], day, st["last"]))
        out.append(Liquidity(name, "sell", st["lo"], day, st["last"]))

    for liq in out:
        for i in range(liq.formed_idx + 1, len(candles)):
            c = candles[i]
            if (liq.side == "buy" and c.high > liq.level) or \
               (liq.side == "sell" and c.low < liq.level):
                liq.swept_idx = i
                break
    return out


def swept_before(liqs: List[Liquidity], idx: int, direction: str,
                 window: int = 60) -> List[Liquidity]:
    """Liquidite balayee dans les `window` bougies precedant `idx`.

    Un OB haussier veut un balayage SELL-side prealable (on purge les stops
    sous un creux avant de repartir a la hausse), et inversement.
    """
    want = "sell" if direction == "bull" else "buy"
    return [l for l in liqs
            if l.side == want and l.swept_idx is not None
            and idx - window <= l.swept_idx <= idx]
