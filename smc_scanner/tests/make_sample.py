"""Genere un CSV M1 synthetique : tendance + replis + sweeps.
Sert aux tests et a la demo. Ce n'est PAS de la donnee de marche."""
import csv, math, random
from datetime import datetime, timedelta, timezone


def gen(path, bars=20000, start=4300.0, seed=7):
    rnd = random.Random(seed)
    t = datetime(2026, 9, 1, tzinfo=timezone.utc)
    px = start
    rows = []
    for i in range(bars):
        # derive lente + cycle + bruit : cree des jambes et des replis
        drift = 0.0009 * math.sin(i / 900.0) + 0.00025
        px *= (1 + drift / 100.0 + rnd.gauss(0, 0.00022))
        o = px
        rng = abs(rnd.gauss(0, 0.45)) + 0.12
        c = o + rnd.gauss(0, 0.30)
        h = max(o, c) + abs(rnd.gauss(0, rng * 0.5))
        l = min(o, c) - abs(rnd.gauss(0, rng * 0.5))
        rows.append([t.strftime("%Y-%m-%d %H:%M:%S"), f"{o:.2f}", f"{h:.2f}",
                     f"{l:.2f}", f"{c:.2f}", rnd.randint(50, 900)])
        px = c
        t += timedelta(minutes=1)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        w.writerows(rows)
    return path


if __name__ == "__main__":
    import sys
    print(gen(sys.argv[1] if len(sys.argv) > 1 else "sample_m1.csv"))
