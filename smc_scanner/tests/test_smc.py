"""Tests du moteur. Le plus important : test_no_lookahead."""
import sys, os, unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smc.model import Candle
from smc.data import resample, atr
from smc.structure import find_pivots, build_structure, momentum
from smc.zones import find_order_blocks, find_fvg, find_volume_gaps, ote_zone
from smc.execution import is_engulfing, is_inside_bar, is_overlap
from smc.sessions import session_levels

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def mk(seq, step=1):
    """seq = [(o,h,l,c), ...] -> bougies espacees de `step` minutes."""
    return [Candle(T0 + timedelta(minutes=i * step), *v) for i, v in enumerate(seq)]


class TestStructure(unittest.TestCase):
    def test_pivot_confirme_apres_coup(self):
        c = mk([(1, 2, 0.5, 1.5), (1.5, 3, 1.4, 2.8), (2.8, 5, 2.7, 4.5),
                (4.5, 4.6, 3, 3.2), (3.2, 3.3, 2, 2.2)])
        ph = [p for p in find_pivots(c, 2, 2) if p.side == "high"]
        self.assertTrue(ph, "un sommet fractal doit etre detecte")
        p = ph[0]
        self.assertEqual(p.idx, 2)
        self.assertEqual(p.known_idx, 4, "le pivot n'est connu que 2 bougies plus tard")

    def test_bos_haussier_sur_cloture(self):
        seq = [(10, 11, 9, 10.5), (10.5, 12, 10, 11.5), (11.5, 14, 11, 13.5),
               (13.5, 13.8, 11.5, 12.0), (12.0, 12.2, 10.5, 10.8),
               (10.8, 11.5, 10.6, 11.4), (11.4, 13.0, 11.3, 12.9),
               (12.9, 15.5, 12.8, 15.4), (15.4, 16, 15, 15.8)]
        ev, _ = build_structure(mk(seq), 2, 2)
        bos = [e for e in ev if e.kind == "BOS" and e.direction == "bull"]
        self.assertTrue(bos, "une cassure haussiere doit produire un BOS")

    def test_momentum_suit_dernier_evenement(self):
        self.assertEqual(momentum([]), "Neutre")


class TestZones(unittest.TestCase):
    def test_fvg_haussier(self):
        c = mk([(10, 11, 9.5, 10.8), (10.8, 14, 10.7, 13.8), (13.8, 15, 12, 14.5)])
        f = find_fvg(c)
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].direction, "bull")
        self.assertAlmostEqual(f[0].low, 11.0)   # haut de la bougie 1
        self.assertAlmostEqual(f[0].high, 12.0)  # bas de la bougie 3

    def test_pas_de_fvg_si_meches_se_croisent(self):
        c = mk([(10, 13, 9.5, 12), (12, 14, 11, 13.5), (13.5, 15, 12.5, 14.5)])
        self.assertEqual(find_fvg(c), [])

    def test_volume_gap_vrai_trou(self):
        c = mk([(10, 11, 9.5, 10.8), (12.5, 13, 12.2, 12.8)])
        vg = find_volume_gaps(c)
        self.assertEqual(len(vg), 1)
        self.assertAlmostEqual(vg[0].low, 11.0)
        self.assertAlmostEqual(vg[0].high, 12.2)

    def test_ote_bornes_strictes(self):
        lo, hi = ote_zone(100.0, 200.0, "bull")
        self.assertAlmostEqual(lo, 121.4, places=1)   # 200 - 0.786*100
        self.assertAlmostEqual(hi, 138.2, places=1)   # 200 - 0.618*100
        lo2, hi2 = ote_zone(100.0, 200.0, "bear")
        self.assertAlmostEqual(lo2, 161.8, places=1)
        self.assertAlmostEqual(hi2, 178.6, places=1)


class TestNoLookahead(unittest.TestCase):
    """Le test qui decide si le scanner est honnete."""

    def _candles(self):
        import random
        rnd = random.Random(3)
        out, px = [], 100.0
        for i in range(600):
            o = px
            c = o + rnd.gauss(0.01, 0.6)
            h = max(o, c) + abs(rnd.gauss(0, 0.3))
            l = min(o, c) - abs(rnd.gauss(0, 0.3))
            out.append(Candle(T0 + timedelta(minutes=i), o, h, l, c))
            px = c
        return out

    def test_ob_jamais_connu_avant_sa_cassure(self):
        c = self._candles()
        ev, _ = build_structure(c, 2, 2)
        for ob in find_order_blocks(c, ev, "M1"):
            self.assertGreater(ob.confirmed_idx, ob.origin_idx,
                               "un OB ne peut etre connu avant la cassure qui le cree")

    def test_scan_tronque_est_un_prefixe_du_scan_complet(self):
        """Scanner les 400 premieres bougies doit donner exactement les memes
        OB que scanner les 600 puis filtrer sur confirmed_idx <= 399.
        Si ce test casse, le moteur lit le futur."""
        c = self._candles()
        cut = 400

        def obs_of(data):
            ev, _ = build_structure(data, 2, 2)
            return find_order_blocks(data, ev, "M1")

        partiel = [(o.origin_idx, o.confirmed_idx, round(o.low, 6), round(o.high, 6))
                   for o in obs_of(c[:cut])]
        complet = [(o.origin_idx, o.confirmed_idx, round(o.low, 6), round(o.high, 6))
                   for o in obs_of(c) if o.confirmed_idx < cut]
        self.assertEqual(partiel, complet)


class TestExecution(unittest.TestCase):
    def test_englobante(self):
        prev = Candle(T0, 10, 10.5, 9.5, 9.8)      # corps 9.8-10
        cur = Candle(T0, 9.6, 11, 9.4, 10.6)       # corps 9.6-10.6 : englobe
        self.assertTrue(is_engulfing(prev, cur, "bull"))
        self.assertFalse(is_engulfing(prev, cur, "bear"))

    def test_inside_bar(self):
        prev = Candle(T0, 10, 12, 9, 11)
        cur = Candle(T0, 10.5, 11.5, 9.5, 10.8)
        self.assertTrue(is_inside_bar(prev, cur))
        self.assertFalse(is_engulfing(prev, cur, "bull"))

    def test_overlap_distinct_des_deux_autres(self):
        prev = Candle(T0, 10, 10.6, 9.9, 10.4)     # corps 10.0-10.4
        cur = Candle(T0, 10.2, 10.9, 9.8, 10.5)    # perce 10.0 par le bas, cloture haut
        self.assertTrue(is_overlap(prev, cur, "bull"))
        self.assertFalse(is_inside_bar(prev, cur))


class TestSessions(unittest.TestCase):
    def test_sessions_et_balayage(self):
        c = [Candle(T0.replace(hour=0) + timedelta(minutes=i * 30),
                    100, 101 + (i == 2) * 5, 99, 100) for i in range(40)]
        liqs = session_levels(c)
        self.assertTrue(any(l.session == "Asie" for l in liqs))
        self.assertTrue(all(l.formed_idx >= 0 for l in liqs))


class TestResample(unittest.TestCase):
    def test_m1_vers_m5_aligne(self):
        c = mk([(i, i + 1, i - 1, i + 0.5) for i in range(1, 21)])
        m5 = resample(c, "M5")
        self.assertEqual(len(m5), 4)
        self.assertEqual(m5[0].open, c[0].open)
        self.assertEqual(m5[0].close, c[4].close)
        self.assertEqual(m5[0].high, max(x.high for x in c[:5]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
