#!/usr/bin/env python3
"""
Etude des zones SMC (Order Blocks) : quelles zones reagissent le mieux ?

Pour chaque OB (meme logique que pine/smc_ob_fvg.pine) :
  - entree limite au bord proximal a la 1ere retouche (avant mitigation / expiration)
  - SL = bord distal -/+ 0.1 ATR, TP = RR x risque, sortie temporelle apres --hold barres
  - frais inclus (2 x --fee, en % du prix, convertis en R)
  - SL prioritaire sur TP dans une meme barre (hypothese conservatrice)
Puis : stats par feature (score, FVG, alignement HTF, CHoCH vs continuation, impulsion,
taille de zone, age, session, sens), comparaison a un benchmark d'entrees aleatoires
de meme risque, et validation train/test (selection sur 60 % du temps, test sur 40 %).

Usage :
  python3 smc_zone_study.py --csv data.csv            # CSV OHLC (time,open,high,low,close)
  python3 smc_zone_study.py --binance BTCUSDT --interval 1h --years 3
  python3 smc_zone_study.py --synthetic               # test de plomberie (marche aleatoire)
"""
import argparse
import itertools
import json
import math
import sys
import time
import urllib.request

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- data
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next(c for c in ("time", "timestamp", "datetime", "date", "open time") if c in df.columns)
    t = df[tcol]
    if pd.api.types.is_numeric_dtype(t):
        unit = "ms" if t.iloc[0] > 1e11 else "s"
        idx = pd.to_datetime(t, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(t, utc=True)
    out = df[["open", "high", "low", "close"]].astype(float)
    out.index = idx
    return out.sort_index()[~out.index.duplicated()]


def load_binance(symbol, interval, years):
    ms = {"1m": 6e4, "5m": 3e5, "15m": 9e5, "1h": 3.6e6, "4h": 1.44e7, "1d": 8.64e7}[interval]
    end = int(time.time() * 1000)
    start = int(end - years * 365 * 86400 * 1000)
    rows = []
    while start < end:
        url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}"
               f"&startTime={start}&limit=1000")
        with urllib.request.urlopen(url, timeout=30) as r:
            chunk = json.loads(r.read())
        if not chunk:
            break
        rows += chunk
        start = int(chunk[-1][0] + ms)
    df = pd.DataFrame(rows).iloc[:, :5]
    df.columns = ["t", "open", "high", "low", "close"]
    df.index = pd.to_datetime(df.pop("t"), unit="ms", utc=True)
    return df.astype(float)


def synthetic(n=40000, seed=1):
    rng = np.random.default_rng(seed)
    vol = np.empty(n)
    v = 0.002
    for i in range(n):
        v = 0.0001 + 0.9 * v + 0.2 * abs(rng.normal(0, 0.002))
        vol[i] = v
    r = rng.normal(0, 1, n) * vol
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    spread = np.abs(rng.normal(0, 1, (n, 2))) * vol[:, None] * c[:, None]
    h = np.maximum(o, c) + spread[:, 0]
    l = np.minimum(o, c) - spread[:, 1]
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


# ----------------------------------------------------------------- SMC
def wilder_atr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def scan(h, l, c, atr, L, max_scan=400):
    """Structure + Order Blocks, sans lookahead (pivot confirme L barres plus tard)."""
    n = len(c)
    trend = np.zeros(n, dtype=int)
    zones = []
    hi_lvl = lo_lvl = None
    hi_bar = lo_bar = 0
    hi_done = lo_done = True
    tr = 0
    for i in range(n):
        k = i - L
        if k >= L:
            if h[k] >= h[k - L:i + 1].max():
                hi_lvl, hi_bar, hi_done = h[k], k, False
            if l[k] <= l[k - L:i + 1].min():
                lo_lvl, lo_bar, lo_done = l[k], k, False
        if hi_lvl is not None and not hi_done and c[i] > hi_lvl:
            hi_done = True
            s = max(hi_bar, i - max_scan)
            ob = s + int(np.argmin(l[s:i + 1]))
            zones.append(dict(dir=1, i=i, ob=ob, top=h[ob], bot=l[ob], choch=int(tr == -1),
                              impulse=(h[ob:i + 1].max() - l[ob]) / atr[i]))
            tr = 1
        if lo_lvl is not None and not lo_done and c[i] < lo_lvl:
            lo_done = True
            s = max(lo_bar, i - max_scan)
            ob = s + int(np.argmax(h[s:i + 1]))
            zones.append(dict(dir=-1, i=i, ob=ob, top=h[ob], bot=l[ob], choch=int(tr == 1),
                              impulse=(h[ob] - l[ob:i + 1].min()) / atr[i]))
            tr = -1
        trend[i] = tr
    return zones, trend


def htf_trend(df, rule, L):
    ohlc = df.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    a = wilder_atr(ohlc.high.to_numpy(), ohlc.low.to_numpy(), ohlc.close.to_numpy())
    _, t = scan(ohlc.high.to_numpy(), ohlc.low.to_numpy(), ohlc.close.to_numpy(), a, L)
    s = pd.Series(t, index=ohlc.index).shift(1)  # barre HTF cloturee uniquement
    return s.reindex(df.index, method="ffill").fillna(0).to_numpy().astype(int)


def simulate(h, l, c, start, entry, risk, d, rr, hold, fee_r):
    """SL prioritaire sur TP. Retourne (R net, MFE en R)."""
    sl = entry - d * risk
    tp = entry + d * rr * risk
    end = min(start + hold, len(c))
    mfe = 0.0
    for m in range(start, end):
        hit_sl = (l[m] <= sl) if d == 1 else (h[m] >= sl)
        hit_tp = (h[m] >= tp) if d == 1 else (l[m] <= tp)
        fav = ((h[m] - entry) if d == 1 else (entry - l[m])) / risk
        if hit_sl:
            return -1.0 - fee_r, mfe
        mfe = max(mfe, fav)
        if hit_tp:
            return rr - fee_r, mfe
    last = c[end - 1]
    return d * (last - entry) / risk - fee_r, mfe


def study(df, L, htf, rr, hold, max_age, fee, fvg_min, min_risk_atr, seed):
    o = df.open.to_numpy(); h = df.high.to_numpy(); l = df.low.to_numpy(); c = df.close.to_numpy()
    n = len(c)
    atr = wilder_atr(h, l, c)
    zones, _ = scan(h, l, c, atr, L)
    ht = htf_trend(df, htf, L)

    # FVG cumules (pour savoir si un FVG de meme sens s'est forme dans l'impulsion)
    bull = np.zeros(n, bool); bear = np.zeros(n, bool)
    for k in range(2, n):
        if l[k] > h[k - 2] and c[k - 1] > h[k - 2] and (l[k] - h[k - 2]) > fvg_min * atr[k]:
            bull[k] = True
        if h[k] < l[k - 2] and c[k - 1] < l[k - 2] and (l[k - 2] - h[k]) > fvg_min * atr[k]:
            bear[k] = True
    cb, cs = np.cumsum(bull), np.cumsum(bear)

    rng = np.random.default_rng(seed)
    hours = df.index.hour.to_numpy()
    rows = []
    for z in zones:
        d, i = z["dir"], z["i"]
        top, bot = z["top"], z["bot"]
        touch = None
        for j in range(i + 1, min(i + 1 + max_age, n)):
            # pas de test de cloture ici : conditionner sur c[j] = lookahead (une cloture sous la zone
            # est un trade perdant, pas un trade ignore)
            if (d == 1 and l[j] <= top) or (d == -1 and h[j] >= bot):
                touch = j
                break
        if touch is None:
            continue
        j = touch
        entry = min(top, o[j]) if d == 1 else max(bot, o[j])
        sl = bot - 0.1 * atr[i] if d == 1 else top + 0.1 * atr[i]
        risk = abs(entry - sl)
        gap_through = (d == 1 and entry <= sl) or (d == -1 and entry >= sl)
        if not gap_through and risk < min_risk_atr * atr[i]:
            continue
        if gap_through:  # ouverture au-dela du SL : perte comptee (-1R), jamais ignoree
            risk = max(abs(top - bot) + 0.1 * atr[i], 1e-12)
            fee_r = 2 * fee * entry / risk
            r, mfe = -1.0 - fee_r, 0.0
        else:
            fee_r = 2 * fee * entry / risk
            r, mfe = simulate(h, l, c, j, entry, risk, d, rr, hold, fee_r)

        # placebo : meme mecanique d'ordre limite (meme distance au prix en ATR, meme taille de zone,
        # meme sens), pose a une barre aleatoire -> isole l'effet "zone SMC" de l'effet "mecanique de fill"
        r_rand = np.nan
        for _ in range(5):
            rb = int(rng.integers(L * 3, n - hold - max_age - 2))
            dist = (c[i] - top) / atr[i] if d == 1 else (bot - c[i]) / atr[i]
            lvl = c[rb] - d * max(dist, 0.0) * atr[rb]
            zsz = (abs(top - bot) + 0.1 * atr[i]) / atr[i] * atr[rb]
            jb = next((m for m in range(rb + 1, rb + 1 + max_age)
                       if (d == 1 and l[m] <= lvl) or (d == -1 and h[m] >= lvl)), None)
            if jb is None:
                continue
            eb = min(lvl, o[jb]) if d == 1 else max(lvl, o[jb])
            slb = lvl - d * zsz
            rk = abs(eb - slb)
            if (d == 1 and eb <= slb) or (d == -1 and eb >= slb):
                r_rand = -1.0 - 2 * fee * eb / zsz
            elif rk >= min_risk_atr * atr[rb]:
                r_rand, _m = simulate(h, l, c, jb, eb, rk, d, rr, hold, 2 * fee * eb / rk)
            else:
                continue
            break

        fvg = (cb[i] - cb[z["ob"]] > 0) if d == 1 else (cs[i] - cs[z["ob"]] > 0)
        htf_al = int(ht[i] == d)
        stars = 1 + int(fvg) + htf_al
        hr = hours[j]
        sess = "Asie" if hr < 7 else "Londres" if hr < 13 else "NY" if hr < 21 else "Off"
        rows.append(dict(time=df.index[i], dir="long" if d == 1 else "short", R=r, MFE=mfe, R_rand=r_rand,
                         stars=stars, fvg=int(fvg), htf_align=htf_al, choch=z["choch"],
                         impulse=z["impulse"], size_atr=risk / atr[i], age=j - i, session=sess))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- reporting
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def summarize(g, col="R"):
    n = len(g)
    if n == 0:
        return dict(n=0, win=np.nan, ci="", avgR=np.nan, t=np.nan)
    w = int((g[col] > 0).sum())
    lo, hi = wilson(w, n)
    sd = g[col].std(ddof=1) if n > 1 else np.nan
    return dict(n=n, win=100 * w / n, ci=f"[{100*lo:.0f}-{100*hi:.0f}]", avgR=g[col].mean(),
                t=g[col].mean() / (sd / math.sqrt(n)) if sd and sd > 0 else np.nan)


def bucketize(d):
    d = d.copy()
    d["impulse_q"] = pd.qcut(d.impulse, 4, labels=["Q1 faible", "Q2", "Q3", "Q4 fort"], duplicates="drop")
    d["size_q"] = pd.qcut(d.size_atr, 4, labels=["Q1 serree", "Q2", "Q3", "Q4 large"], duplicates="drop")
    d["age_q"] = pd.qcut(d.age, 4, labels=["Q1 retouche rapide", "Q2", "Q3", "Q4 retouche tardive"], duplicates="drop")
    return d


FEATURES = ["stars", "fvg", "htf_align", "choch", "dir", "session", "impulse_q", "size_q", "age_q"]


def table(d, feat, split):
    out = []
    for v, g in d.groupby(feat, observed=True):
        s = summarize(g)
        tr = summarize(g[g.time < split])
        te = summarize(g[g.time >= split])
        out.append(dict(feature=feat, valeur=str(v), n=s["n"], win_pct=round(s["win"], 1), IC95=s["ci"],
                        avgR=round(s["avgR"], 3), t=round(s["t"], 2),
                        avgR_train=round(tr["avgR"], 3), n_tr=tr["n"],
                        avgR_test=round(te["avgR"], 3), n_te=te["n"]))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--binance")
    src.add_argument("--synthetic", action="store_true")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--years", type=float, default=3)
    ap.add_argument("--swing", type=int, default=5)
    ap.add_argument("--htf", default="4h")
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--hold", type=int, default=100)
    ap.add_argument("--max-age", type=int, default=300)
    ap.add_argument("--fee", type=float, default=0.0005, help="par cote, fraction (0.0005 = 0.05 %%)")
    ap.add_argument("--fvg-min", type=float, default=0.3)
    ap.add_argument("--min-risk-atr", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out")
    a = ap.parse_args()

    df = load_csv(a.csv) if a.csv else load_binance(a.binance, a.interval, a.years) if a.binance else synthetic()
    print(f"# Donnees : {len(df)} barres, {df.index[0]} -> {df.index[-1]}")
    d = study(df, a.swing, a.htf, a.rr, a.hold, a.max_age, a.fee, a.fvg_min, a.min_risk_atr, a.seed)
    if len(d) < 50:
        sys.exit(f"Trop peu de trades ({len(d)}) pour conclure.")
    d = bucketize(d.dropna(subset=["R_rand"]).reset_index(drop=True))
    split = d.time.sort_values().iloc[int(len(d) * 0.6)]
    if a.out:
        d.to_csv(a.out, index=False)

    s, b = summarize(d), summarize(d, "R_rand")
    print(f"\n## Global (RR {a.rr}, frais {2*a.fee*100:.2f} % A/R inclus)")
    print(f"- OB retestes : n={s['n']}  win={s['win']:.1f}% {s['ci']}  avgR={s['avgR']:+.3f}  t={s['t']:.2f}")
    print(f"- Placebo (meme ordre limite, niveau aleatoire) : win={b['win']:.1f}%  avgR={b['avgR']:+.3f}  t={b['t']:.2f}")
    print(f"- Edge OB vs aleatoire : {s['avgR'] - b['avgR']:+.3f} R/trade")
    print(f"- Split train/test a {split}")

    full = pd.concat([table(d, f, split) for f in FEATURES], ignore_index=True)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n## Par feature (avgR en R net de frais)")
    print(full.to_string(index=False))

    # Selection sur train, verification sur test (combinaisons de 2 features binaires/categorielles)
    print("\n## Selection sur TRAIN -> validation sur TEST (combinaisons de 2 features)")
    cands = []
    for f1, f2 in itertools.combinations(FEATURES, 2):
        for (v1, v2), g in d.groupby([f1, f2], observed=True):
            tr, te = g[g.time < split], g[g.time >= split]
            if len(tr) >= 30 and len(te) >= 15:
                cands.append((summarize(tr)["avgR"], f"{f1}={v1} & {f2}={v2}", len(tr), summarize(te)["avgR"], len(te)))
    cands.sort(reverse=True)
    print(f"{len(cands)} combinaisons testees -> attendre du hasard parmi les meilleures (comparaison multiple)")
    for tr_r, name, ntr, te_r, nte in cands[:8]:
        print(f"- {name:55s} train {tr_r:+.3f} (n={ntr})  ->  test {te_r:+.3f} (n={nte})")
    if cands:
        top = cands[:8]
        print(f"- Moyenne test des 8 meilleures : {np.mean([t[3] for t in top]):+.3f} R "
              f"(si ~0 ou < 0 : surapprentissage, pas d'edge)")


if __name__ == "__main__":
    main()
