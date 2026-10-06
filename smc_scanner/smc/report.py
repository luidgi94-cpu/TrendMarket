"""Rendu du rapport au format demande."""
from typing import List, Dict
from .model import Candle, Liquidity
from .execution import approach_state

RULE = "=" * 78


def _fmt(p: float) -> str:
    return f"{p:,.2f}".replace(",", " ")


def render(symbol: str, tfs: List[str], momentum: Dict[str, str],
           liqs: List[Liquidity], ranked: List[Dict], base: List[Candle],
           atr_ref: float, max_obs: int = 15, price_decimals: int = 2) -> str:
    last = base[-1]
    L: List[str] = []
    L.append(RULE)
    L.append(f"  RAPPORT SMC / ICT  --  {symbol}")
    L.append(f"  Derniere bougie : {last.ts:%Y-%m-%d %H:%M} UTC   Cloture : {_fmt(last.close)}")
    L.append(RULE)

    L.append("\n1. ETAT DU MOMENTUM PAR TIMEFRAME")
    L.append("-" * 78)
    for tf in tfs:
        L.append(f"   {tf:<5} : {momentum.get(tf, 'Neutre')}")

    L.append("\n2. CARTOGRAPHIE DES SESSIONS (bassins de liquidite)")
    L.append("-" * 78)
    if not liqs:
        L.append("   Aucune session complete dans les donnees fournies.")
    else:
        days = sorted({l.day for l in liqs})[-2:]
        L.append(f"   {'Jour':<12}{'Session':<10}{'Cote':<12}{'Niveau':>12}   Statut")
        for d in days:
            for l in [x for x in liqs if x.day == d]:
                side = "Buy-side" if l.side == "buy" else "Sell-side"
                st = f"BALAYEE @{l.swept_idx}" if l.swept else "INTACTE"
                L.append(f"   {d:<12}{l.session:<10}{side:<12}{_fmt(l.level):>12}   {st}")

    L.append("\n3. ORDER BLOCKS DETECTES (classes par pertinence)")
    L.append("-" * 78)
    if not ranked:
        L.append("   Aucun OB ne passe les filtres sur la periode fournie.")
    else:
        for r in ranked[:max_obs]:
            ob = r["ob"]
            sens = "Bullish" if ob.direction == "bull" else "Bearish"
            L.append(f"\n   [{r['rank']}] OB {sens} {ob.tf}  "
                     f"{_fmt(ob.low)} - {_fmt(ob.high)}")
            L.append(f"        Forme le      : {ob.ts:%Y-%m-%d %H:%M} UTC")
            L.append(f"        Confirme par  : {ob.meta['event']} "
                     f"(niveau {_fmt(ob.meta['break_level'])}) "
                     f"| impulsion {ob.meta['impulse_atr']} x ATR")
            L.append(f"        Mitige        : {'OUI @' + str(ob.mitigated_idx) if ob.mitigated else 'NON'}")
            L.append(f"        Zone OTE      : {'OUI' if r['ote'] else 'non'}")
            if r["fvg"]:
                f = r["fvg"]
                L.append(f"        FVG lie       : {_fmt(f.low)} - {_fmt(f.high)} ({f.tf})")
            else:
                L.append("        FVG lie       : aucun a proximite")
            if r["sweeps"]:
                s = ", ".join(f"{x.session} {x.side}-side {_fmt(x.level)}" for x in r["sweeps"][:3])
                L.append(f"        Liquidite     : balayage prealable -> {s}")
            else:
                L.append("        Liquidite     : pas de balayage de session prealable")
            L.append(f"        Position prix : {approach_state(ob, last, atr_ref)}")

    L.append("\n4. PLAN DE CONFIRMATION")
    L.append("-" * 78)
    live = [r for r in ranked if r["rank"] in ("A+", "A")
            and approach_state(r["ob"], last, atr_ref) in ("DANS LA ZONE", "APPROCHE (< 1 ATR)")]
    if not live:
        L.append("   Aucun OB de rang A+/A a portee immediate du prix.")
        L.append("   -> Pas de protocole arme. Surveillance uniquement.")
    else:
        for r in live:
            ob = r["ob"]
            sens = "BUY" if ob.direction == "bull" else "SELL"
            L.append(f"   Le prix est en {approach_state(ob, last, atr_ref)} de l'OB "
                     f"{ob.tf} [{r['rank']}] {_fmt(ob.low)}-{_fmt(ob.high)}.")
            L.append(f"   -> Bascule analyse sur M5, lecture de la reaction.")
            L.append(f"   -> Gachette M1 attendue : Englobante ou Chevauchante, sens {sens}.")
            L.append(f"   -> Si validee : ordre limite {sens} sur la zone {ob.tf} d'origine,")
            L.append(f"      invalidation sous {_fmt(ob.low)}." if ob.direction == "bull"
                     else f"      invalidation au-dessus de {_fmt(ob.high)}.")
    L.append("\n" + RULE)
    return "\n".join(L)
