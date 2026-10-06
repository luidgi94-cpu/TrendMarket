# Scanner SMC / ICT

Implémentation exécutable du protocole d'analyse Order Blocks / structure /
liquidité de sessions, en Python stdlib pur (aucune dépendance).

## Pourquoi du code plutôt qu'une lecture visuelle

Un scanner qui lit le graphique **après coup** paraît excellent et échoue en
live. Ce moteur date chaque zone à l'instant où elle devient *connaissable* :
un Order Block n'existe qu'à la clôture de la bougie qui casse la structure,
pas à la bougie qui le forme. `tests/test_smc.py::TestNoLookahead` vérifie
que scanner les N premières bougies donne exactement le même résultat que
scanner tout l'historique puis filtrer. Si ce test casse, le moteur lit le
futur et tout backtest bâti dessus est faux.

## Usage

```bash
python3 -m smc.scan donnees_m1.csv --symbol XAUUSD
python3 -m smc.scan donnees_m1.csv --ob-tfs H1,M15 --min-impulse-atr 1.5
```

CSV attendu : en-tête avec `time,open,high,low,close[,volume]`.
Horodatages sans fuseau traités comme UTC.

Données de démonstration (synthétiques, **pas** du marché) :

```bash
python3 tests/make_sample.py sample_m1.csv
python3 -m smc.scan sample_m1.csv --symbol DEMO
```

## Définitions implémentées

| Élément | Règle codée |
|---|---|
| BOS | clôture de corps au-delà du dernier sommet/creux majeur, dans le sens de la tendance |
| CHoCH | clôture sous le dernier creux ayant entraîné le plus haut (et inverse) |
| Order Block | dernière bougie opposée avant l'impulsion qui casse, déplacement ≥ `min_impulse_atr` × ATR |
| FVG | 3 bougies, mèches 1 et 3 disjointes |
| Volume Gap | aucun recouvrement entre deux bougies consécutives |
| OTE | retracement strict 0.618 – 0.786 de la jambe impulsive |
| Sessions | H/L Asie 00–08, Londres 07–16, NY 12–21 UTC (configurable dans `smc/sessions.py`) |
| Rangs | A+ : non mitigé + OTE + FVG adjacent + balayage de liquidité préalable — A : non mitigé + FVG + aligné HTF — B : le reste |
| Gâchette M1 | Englobante / Inside Bar / Lap, distinguées (`smc/execution.py`) |

## Tests

```bash
python3 tests/test_smc.py
```

## Limites assumées

- Le moteur **détecte et classe**. Il ne dit pas si ces zones ont une espérance
  positive : ça, seul un backtest avec coûts le dira.
- Aucun dimensionnement de position n'est fourni — à décider hors de cet outil.
- Le Volume Gap est quasi inexistant sur le FX/or hors week-end : attendre
  ~1 détection par semaine, pas davantage.
