"""Classement A+ / A / B des Order Blocks selon les criteres fournis."""
from typing import List, Dict
from .model import Candle, Zone, Liquidity
from .zones import in_ote, adjacent_fvg
from .sessions import swept_before


def rank_order_blocks(obs: List[Zone], candles: List[Candle], fvgs: List[Zone],
                      liqs: List[Liquidity], htf_momentum: str,
                      fvg_tol: float, sweep_window: int = 60) -> List[Dict]:
    """A+ : non mitige + en OTE + FVG adjacent + balayage de liquidite prealable
       A  : non mitige + FVG present + aligne au momentum du TF superieur
       B  : mitige, ou hors OTE, ou sans FVG a proximite
    """
    out: List[Dict] = []
    for ob in obs:
        fvg = adjacent_fvg(ob, fvgs, fvg_tol)
        ote = in_ote(ob, candles)
        sweeps = swept_before(liqs, ob.confirmed_idx, ob.direction, sweep_window)
        aligned = (htf_momentum == "Haussier" and ob.direction == "bull") or \
                  (htf_momentum == "Baissier" and ob.direction == "bear")

        if not ob.mitigated and ote and fvg is not None and sweeps:
            rank = "A+"
        elif not ob.mitigated and fvg is not None and aligned:
            rank = "A"
        else:
            rank = "B"

        out.append({
            "ob": ob, "rank": rank, "fvg": fvg, "ote": ote,
            "sweeps": sweeps, "aligned": aligned,
        })
    order = {"A+": 0, "A": 1, "B": 2}
    out.sort(key=lambda d: (order[d["rank"]], -d["ob"].confirmed_idx))
    return out
