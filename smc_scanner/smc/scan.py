"""Point d'entree CLI : construit le rapport multi-timeframe."""
import argparse
import sys
from typing import List
from .data import load_csv, resample, atr, TF_MINUTES
from .structure import build_structure, momentum
from .zones import find_order_blocks, find_fvg, find_volume_gaps
from .sessions import session_levels
from .ranking import rank_order_blocks
from .report import render

DEFAULT_TFS = ["H1", "M30", "M15", "M5", "M1"]


def scan(candles, symbol="INSTRUMENT", tfs=None, ob_tfs=("H1", "M15"),
         min_impulse_atr=1.5, pivot_left=2, pivot_right=2, sweep_window=60):
    tfs = tfs or DEFAULT_TFS
    series = {}
    for tf in tfs:
        s = resample(candles, tf)
        if len(s) >= 4 * (pivot_left + pivot_right + 1):
            series[tf] = s

    if not series:
        raise SystemExit("Pas assez de donnees pour construire un seul timeframe.")

    moms, events = {}, {}
    for tf, s in series.items():
        ev, _ = build_structure(s, pivot_left, pivot_right)
        events[tf] = ev
        moms[tf] = momentum(ev)

    htf = next((t for t in ("H1", "M30", "M15") if t in moms), list(moms)[0])
    base_tf = min(series, key=lambda t: TF_MINUTES[t])
    base = series[base_tf]
    liqs = session_levels(base)

    ranked = []
    for tf in ob_tfs:
        if tf not in series:
            continue
        s = series[tf]
        a = atr(s)
        tol = a[-1] * 0.5 if a and a[-1] > 0 else 0.0
        obs = find_order_blocks(s, events[tf], tf, min_impulse_atr=min_impulse_atr)
        fvgs = find_fvg(s, tf)
        # la liquidite est indexee sur le TF de base : on la reechantillonne
        liq_tf = session_levels(s)
        ranked += rank_order_blocks(obs, s, fvgs, liq_tf, moms[htf], tol, sweep_window)

    order = {"A+": 0, "A": 1, "B": 2}
    ranked.sort(key=lambda d: (order[d["rank"]], -d["ob"].confirmed_idx))

    a_base = atr(base)
    return render(symbol, [t for t in tfs if t in series], moms, liqs,
                  ranked, base, a_base[-1] if a_base else 0.0)


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(description="Scanner SMC/ICT multi-timeframe")
    p.add_argument("csv", help="CSV OHLC (la plus petite UT disponible, ex. M1)")
    p.add_argument("--symbol", default="INSTRUMENT")
    p.add_argument("--tfs", default=",".join(DEFAULT_TFS))
    p.add_argument("--ob-tfs", default="H1,M15", help="TF ou chercher les OB")
    p.add_argument("--min-impulse-atr", type=float, default=1.5)
    p.add_argument("--pivot", type=int, default=2, help="bougies gauche/droite du fractal")
    p.add_argument("--sweep-window", type=int, default=60)
    a = p.parse_args(argv)

    candles = load_csv(a.csv)
    if not candles:
        print("Aucune bougie lue.", file=sys.stderr)
        return 1
    print(scan(candles, a.symbol, a.tfs.split(","), tuple(a.ob_tfs.split(",")),
               a.min_impulse_atr, a.pivot, a.pivot, a.sweep_window))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
