# Systeme de trading swing — BTC / XAU + diversification

Systeme de suivi de tendance multi-actifs, positions tenues plusieurs semaines
a plusieurs mois. Pas d'intraday, pas de scalping.

## Pourquoi cette approche

L'objectif n'est **pas** le winrate. Le winrate est un parametre libre : en
resserrant le take-profit on atteint 90% de trades gagnants et une esperance
negative. C'est le profil de risque qui a detruit XIV en fevrier 2018.

L'objectif est l'**esperance nette de couts par unite de risque**, avec un
**skew positif** : beaucoup de petites pertes coupees vite, quelques trends
tenus longtemps. Ce systeme affiche un winrate de 25-45% par construction.
C'est voulu.

Trois raisons de choisir le trend-following plutot qu'autre chose :

- **Skew positif** — on survit aux crises au lieu de les subir.
- **Robustesse documentee** — plus de 100 ans d'historique, toutes classes
  d'actifs. Ce n'est pas un artefact d'over-fitting.
- **Couts negligeables** — detention de plusieurs semaines, le spread ne
  mange pas l'edge (contrairement au scalping).

## Architecture

| Module | Role |
|---|---|
| `quant/config.py` | Univers, couts par instrument, parametres de risque |
| `quant/signals.py` | Ensemble de forecasts : EWMAC 3 horizons + Donchian 3 horizons |
| `quant/portfolio.py` | Vol targeting, IDM, plafonds, buffering |
| `quant/overlays.py` | Funding perp (BTC/ETH), macro DXY/taux (XAU/XAG) |
| `quant/backtest.py` | Moteur vectorise, net de couts, sans lookahead |
| `quant/metrics.py` | Sharpe, Sortino, DD, skew, winrate, esperance en R |
| `quant/montecarlo.py` | Comparaison d'univers sur N chemins independants |
| `quant/data.py` | Yahoo Finance / Binance funding / CSV, avec cache |
| `quant/synthetic.py` | Donnees simulees pour valider le moteur |

### Decisions de conception

**Aucun parametre n'est optimise sur les donnees.** Les horizons (16/64,
32/128, 64/256 et Donchian 60/120/240) viennent de la litterature. Sur
2 instruments, toute optimisation produit de l'over-fitting garanti.

**Forecast continu, pas binaire.** Un trend faible donne une petite position.
C'est ce qui evite de se faire hacher sur les faux departs.

**Vol targeting obligatoire.** BTC a une vol ~50%, l'or ~16%. Sans
normalisation par l'ATR, BTC represente ~75% du risque du book et l'or n'est
que de la decoration.

**IDM (multiplicateur de diversification).** `1/sqrt(w'Cw)` sur correlation
glissante. Il quantifie directement combien on peut lever le book parce que
les positions ne bougent pas ensemble. IDM 1.0 = aucune diversification.

**Buffering par instrument.** Bande morte proportionnelle a la taille propre
de chaque instrument. Reduit le turnover de ~20-30% sans degrader le signal.

**Les overlays ne creent jamais un trade**, ils modulent la taille d'un trade
que le trend a deja valide. Chaque filtre autorise a *ouvrir* une position
multiplie les chemins d'over-fitting.

## Validation du moteur

```bash
python quant/tests/test_engine.py
```

Le test decisif est `test_no_edge_on_random_walk` : sur une marche aleatoire
sans structure exploitable, le systeme doit produire un Sharpe ~0. S'il gagne,
il y a du lookahead dans le code et tout le reste est invalide.

Resultat actuel : **Sharpe moyen 0.00 sur 8 marches aleatoires**. Le moteur
est propre.

## Resultats

### Donnees synthetiques — Monte Carlo, 60 chemins de 11.7 ans

Le monde simule est calibre pour qu'un **oracle parfait plafonne a Sharpe
0.40** (persistance de trend ~200 jours, conforme a la litterature momentum).

| Univers | Sharpe median | P(decennie perdante) | max DD median | IDM |
|---|---|---|---|---|
| BTC + XAU | 0.13 | **40%** | -34.3% | 1.40 |
| + ETH + XAG | 0.21 | 37% | -30.4% | 1.73 |
| 8 instruments | **0.27** | **22%** | **-23.0%** | 2.46 |

Test apparie (meme chemin de marche, deux univers) :

- Sharpe : **+0.115** en passant de 2 a 8 instruments, **p = 0.008**
- Max drawdown : **+10.7 points** de DD en moins, **p < 0.001**, ameliore sur
  **78%** des chemins

**Lecture** : avec 2 actifs seulement, une chance sur deux et demie de
traverser une decennie entiere en perte. Avec 8, une sur cinq. Le gain ne
vient pas d'un meilleur signal — c'est exactement le meme code — mais
uniquement de la diversification.

### Donnees reelles

Non mesurees ici : l'environnement d'execution bloque l'acces reseau aux
fournisseurs de donnees de marche. **Les chiffres ci-dessus valident le
moteur, pas un edge.** Seul un backtest sur donnees reelles peut trancher.

## Utilisation

```bash
pip install -r requirements.txt

python quant/tests/test_engine.py      # valider le moteur d'abord
python run_backtest.py --synthetic     # demo hors-ligne
python run_backtest.py --mc 60         # Monte Carlo comparatif
python run_backtest.py                 # donnees reelles (reseau requis)
python run_backtest.py --refresh       # forcer le retelechargement
```

Sans reseau, deposer des CSV (`date,value`) dans `data_cache/` :
`BTC.csv`, `XAU.csv`, etc. Export TradingView ou broker directement utilisable.

## Attentes realistes

Sur donnees reelles, avec 8 instruments et vol cible 12% :

| | Attendu |
|---|---|
| Sharpe | 0.6 - 0.9 |
| Winrate par trade | 30 - 45% |
| Max drawdown | 15 - 25% |
| Duree de detention mediane | 2 - 3 mois |
| Periode sous l'eau | jusqu'a 18 mois |

Le risque principal de ce systeme n'est pas qu'il perde de l'argent. C'est
qu'on l'abandonne pendant les 18 mois sous l'eau.

## Limites connues

- **Overlay funding non valide** — le code est en place mais l'historique
  Binance n'a pas pu etre telecharge ici.
- **Overlay macro or approximatif** — `^TNX` est un taux nominal, pas reel.
  Utiliser un ETF TIPS ou une serie FRED pour le taux reel.
- **Pas de blackout news** sur donnees quotidiennes. Pertinent surtout en live.
- **Capacite non modelisee.** A taille retail, non contraignant.
