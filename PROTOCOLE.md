# Protocole d'execution

Fige le 16 septembre 2026, avant tout engagement de capital reel.

Ce document est pre-enregistre : il est ecrit AVANT de trader, pour que les
limites ne soient pas renegociees au moment ou elles feront mal. C'est la
seule raison d'etre d'un protocole. Un plan de risque redige apres une
serie de pertes n'est pas un plan de risque, c'est une rationalisation.

---

## 1. Ce qui a ete mesure, et ce qui ne l'a pas ete

Sur 527 jours de XAUUSD reel, export MetaTrader 5 du broker :

    2 070 trades | 3.93 par jour | winrate 57.4%
    esperance +0.178 R brute | +0.115 R nette de frais
    temoin par permutation +0.063 R | ecart +0.115 R | p = 0.0001
    ecart type par trade 1.19 R
    drawdown maximum observe -49.7%

Frais modelises : 0.23 USD de spread + 0.10 USD de slippage, aller-retour,
par once. Sur un stop median de 5 points, cela represente 0.066 R, soit
**36% de l'esperance brute**.

Ce qui N'A PAS ete mesure, et qu'il ne faut donc pas supposer acquis :

- la variante **sans prise partielle**, qui est pourtant la seule
  praticable a 0.01 lot (voir section 3) ;
- l'effet du **filtre de largeur de stop** a 7 points ;
- le comportement sous un spread superieur a 0.23, alors que le spread
  reel s'elargit a l'ouverture de Londres et autour des publications
  macroeconomiques.

Ces trois points sont a mesurer des que les donnees M1 sont de nouveau
disponibles.

---

## 2. Sensibilite au cout : le premier risque

L'edge net vaut +0.115 R. Le cout vaut 0.066 R. Le rapport entre les deux
est le seul chiffre qui decide de la rentabilite de ce systeme.

| Spread moyen | Cout en R | Edge net |
|---|---|---|
| 0.23 (modelise) | 0.066 | +0.115 |
| 0.30 | 0.080 | +0.101 |
| 0.35 | 0.090 | **+0.091** |
| 0.50 | 0.120 | +0.061 |

Douze cents de spread supplementaires retirent 21% de l'edge. Avant toute
autre optimisation, verifier le spread reellement paye sur le releve du
courtier, pas celui affiche dans la plateforme.

---

## 3. Contrainte de taille : la prise partielle est impossible

Le systeme valide ferme 33% de la position au TP1 puis remonte le stop a
l'equilibre. Le lot minimum du courtier est 0.01. Il n'existe pas de
fraction de 0.01 lot.

| | 0.01 lot | 0.02 lot |
|---|---|---|
| Prise partielle au TP1 | impossible | possible |
| Risque par trade (stop 5 pts) | 5 USD = 1.11% | 10 USD = 2.22% |
| Drawdown extrapole | ~-50% | ~-75% |

**Decision : 0.01 lot, sans prise partielle, sortie unique au TP2 (3 R).**

Passer a 0.02 lot pour recuperer le partiel placerait le compte en risque
de ruine sur une serie defavorable ordinaire. La prise partielle reduit la
variance ; doubler la taille pour l'obtenir l'augmente bien davantage.
L'echange est perdant.

Consequence assumee : le systeme reellement trade n'est pas exactement
celui qui a ete valide. L'ecart doit etre mesure, pas suppose negligeable.

---

## 4. Contrainte d'execution : la tolerance est inferieure au point

Esperance de +0.115 R sur un stop de 5 points = **0.58 point par trade**.

Une entree passee 0.6 point plus haut que le prix de signal annule
**l'integralite** de l'esperance. Pas une part : la totalite.

Il n'existe que deux modes d'execution compatibles avec cette tolerance :

1. Presence devant l'ecran, alerte sonore, ordre passe en moins de cinq
   secondes apres la cloture de la bougie M5 ;
2. Automatisation en Expert Advisor MT5.

Toute execution differee au-dela de quelques secondes transforme un
systeme a esperance positive en systeme a esperance nulle ou negative.
Ce n'est pas une question de discipline, c'est de l'arithmetique.

---

## 5. Procedure, a l'alerte

1. Lire la derniere ligne du tableau du scanner. Si elle indique
   `ATTENTE`, ne rien faire.
2. Relever Entree, SL, TP2.
3. Calculer `|Entree - SL|`. **Au-dela de 7 points, ne pas prendre le
   trade.** Un stop large porte le risque au-dela de ce que 450 USD
   supporte.
4. MetaTrader 5 : ordre au marche, **0.01 lot**, SL et TP aux prix lus.
5. Ne plus intervenir. Pas de mise a l'equilibre manuelle, pas de sortie
   anticipee, pas de deplacement de stop.

Le point 5 n'est pas une preference de style. Chaque intervention
discretionnaire sur une position ouverte introduit un parametre qui n'a
jamais ete teste, et dont l'effet est donc inconnu.

---

## 6. Limites de risque

| Limite | Seuil | Action |
|---|---|---|
| Trades par jour | 3 | on arrete d'en prendre |
| Pertes consecutives | 2 | on arrete pour la journee |
| Perte journaliere | -3% | on arrete pour la journee |
| Perte sur le compte | **-25% (337 USD)** | **arret complet et reexamen** |

Le plafond de 3 trades par jour, contre 10 dans le backtest, borne le
risque journalier a 3.3%.

Le seuil d'arret a -25% merite une justification, parce qu'il coute de
l'esperance. Le drawdown observe sur le backtest atteint -49.7%. Tenir
jusque-la supposerait d'accepter de voir le compte a 226 USD. A ce niveau
de capital, la remontee n'a pratiquement jamais lieu : la taille minimale
de position devient trop grande relativement au capital restant, et le
risque par trade passe mecaniquement de 1.1% a 2.2%. Le systeme se
degrade au pire moment.

On coupe donc a -25%, en sachant que l'on abandonne de l'esperance.
**La survie passe avant l'esperance.**

---

## 7. Critere d'acceptation, pre-enregistre

Pour distinguer +0.115 R de zero a 95% de confiance, avec un ecart type
par trade de 1.19 R :

    n = (1.96 x 1.19 / 0.115)^2 = 411 trades

A 3 trades par jour : **environ 137 jours de bourse, soit sept mois.**

Avant ce volume, les resultats en direct ne permettent aucune conclusion
sur la validite du systeme. Une bonne semaine ne valide rien ; une
mauvaise semaine n'invalide rien.

Sont donc interdits, jusqu'a 400 trades ou jusqu'a l'arret a -25% :

- toute modification d'un parametre du systeme ;
- toute augmentation de la taille de position ;
- toute reprise du backtest dans le but de trouver un reglage meilleur.

Ce dernier interdit est le plus important. Rechercher un meilleur reglage
apres avoir observe les resultats en direct revient a choisir le modele
sur les donnees de test, ce qui detruit toute valeur informative du test.

---

## 8. Ce qui reste a faire avant d'engager du capital reel

1. Relancer le balayage du filtre de force sur les donnees M1 reelles
   (`balayage_force.py`), chaque seuil contre son temoin.
2. Mesurer la variante **sans prise partielle**, seule praticable.
3. Mesurer l'effet du **filtre de stop a 7 points**.
4. Verifier le spread reellement paye sur le releve du courtier.
5. Forward test en demo, taille reelle, procedure reelle, sur au moins
   50 trades, pour mesurer le **slippage d'execution effectif**. Si le
   decalage median depasse 0.5 point, le systeme n'est pas executable a
   la main et il faut automatiser avant d'aller plus loin.
