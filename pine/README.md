# XAUUSD · Order Block Scanner (Pine Script v6)

Indicateur TradingView : structure (BOS/CHoCH), Order Blocks classés A+/A/B,
FVG, liquidité de sessions, biais unité de temps supérieure.

## Installation

1. TradingView → onglet **Pine Editor** (en bas)
2. Coller le contenu de `ob_scanner.pine`
3. **Enregistrer** puis **Ajouter au graphique**

## Les 4 plafonds TradingView, et comment ils sont gérés ici

| Limite | Plafond | Traitement |
|---|---|---|
| Boîtes dessinées | 500 | `max_boxes_count=500` + suppression FIFO des plus anciennes |
| Appels `request.security` | 40 | **2 seulement** (biais HTF), décalés d'une bougie |
| Repeinture | — | tout est évalué à la clôture ; `lookahead_off` sur le HTF |
| Étiquettes | 500 | `max_labels_count=500` + purge FIFO |

## Le point de conception qui compte

Un Order Block n'est tracé **qu'à la clôture de la bougie qui casse la
structure** — jamais au moment où la bougie se forme. C'est l'erreur n°1 des
scanners SMC : marquer l'OB à son origine donne un graphique où toutes les
zones ont « fonctionné », et un indicateur qui échoue en direct.

## Classement

- **A+** : non mitigé + en zone OTE (0.618–0.786) + FVG entre l'OB et la cassure + balayage de liquidité de session dans la fenêtre
- **A** : FVG présent + aligné avec le biais de l'unité supérieure
- **B** : le reste

## Réglages conseillés pour XAUUSD

| Paramètre | M15 | H1 |
|---|---|---|
| Longueur du pivot | 5 | 5 |
| Recherche OB | 20 | 20 |
| Impulsion min (× ATR) | 1.5 | 1.5 |
| Prolongation | 40 | 60 |

Sessions pré-réglées en **UTC**. Si ton broker affiche une autre heure,
décale les trois champs de session — sinon les niveaux de liquidité sont faux.

## Alertes

Une alerte par OB détecté, au format :
`A+ Order Block haussier · 4277.22 – 4283.46`

Créer l'alerte : clic droit sur le graphique → Ajouter une alerte →
condition = OB Scanner → « Any alert() function call ».

## Ce que l'indicateur ne fait pas

Il **détecte et classe**. Il ne dit pas si ces zones ont une espérance
positive, et ne gère aucun dimensionnement de position.
Mesures faites sur 588 732 bougies M1 XAUUSD (janv. 2025 – sept. 2026) :

- **A+ : 0,27/semaine en H1, 0,32/semaine en M15.** Environ un par mois.
- **93 % des OB H1 sont revisités**, délai médian 11 bougies.
